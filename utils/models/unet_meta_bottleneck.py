import torch 
import torch.nn as nn
from collections import OrderedDict


class UNet2D_meta_bottleneck(nn.Module):
    def __init__(self, cfg, in_channels=1, init_features=2, out_channels=1):
        #super(UNet2D_meta_bottleneck, self).__init__()
        stride = 2
        pool = 2
        self.features = int(init_features)
        features= self.features
        self.meta_feats=cfg.MODEL.META_FEATURES
        self.device = cfg.MODEL.DEVICE

        self.encoder1 = UNet2D_meta_bottleneck._block(in_channels, features, 'enc1')
        self.pool1 = nn.MaxPool2d(pool, stride=stride)
        self.encoder2 = UNet2D_meta_bottleneck._block(features, features*2, 'enc2')
        self.pool2 = nn.MaxPool2d(pool, stride=stride)
        self.encoder3 = UNet2D_meta_bottleneck._block(features*2, features*4, 'enc3')
        self.pool3 = nn.MaxPool2d(pool, stride=stride)

        self.trans3 = UNet2D_meta_bottleneck._trans(features*4, 'trans3')
        self.dec3 = UNet2D_meta_bottleneck._block(features*4+features*2,features*2, 'dec3')

        self.trans2 = UNet2D_meta_bottleneck._trans(features*2, 'trans2')
        self.dec2 = UNet2D_meta_bottleneck._block(features*2+features, features, 'dec2')

        self.dec1 = UNet2D_meta_bottleneck._block(features,out_channels, 'dec1')

        self.conv_final = nn.Conv2d(out_channels,out_channels,kernel_size=3, stride=1, padding=1)
        self.final_softmax = nn.Sigmoid().to(self.device)

    def forward(self, x, meta):    
        #encoder
        enc1 = self.encoder1(x.to(self.device))
        _skip_connection_1 = enc1.to(self.device)
        enc1_pool=self.pool2(enc1)

        enc2 = self.encoder2(enc1_pool)
        _skip_connection_2 = enc2.to(self.device)
        enc2_pool=self.pool3(enc2)
        
        enc3 = self.encoder3(enc2_pool).to(self.device)

        # linear layer bottleneck
        #print(enc3.size())
        ##should change the size based on the image size and feature size
        #print(x.size())
        features = self.features
        in_channels = features*4
        out_channels = features*4

        k = 3
        p = 1
        s = 1
        name = 'meta'
        pre_lin = nn.Sequential(OrderedDict(
                [(name+'_1', nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=k, stride = s, padding=p, bias=False)),
                 (name+'relu1',nn.ReLU(inplace=True)),
                 (name+'bn1', nn.BatchNorm2d(out_channels)),
                ])).to(self.device)
        
        xx=pre_lin(enc3).to(self.device)
        #print(xx.shape)
        batch_size = 4
        chan = self.features*4

        feat = (chan, batch_size, 10, 32, 32)

        feats = chan*10*32*32
        xx=xx.view(-1,feat[1])
        #print(xx.size())
        
        meta_shape = meta.shape[0]
        meta_flat=meta.view(-1,meta_shape)
        xx=torch.cat((xx,meta_flat),dim=0)
        #print(xx.shape)
        xx=xx.view(xx.size(1), -1).to(self.device)
        #print(xx.shape) 
        

        in_features_lin = feats + self.meta_feats
        _out_features_lin = self.meta_feats
        out_features_lin = feats
        #print('out_feats:', out_features_lin)

        lin = nn.Sequential(OrderedDict(
                [(name+'_1_linear', nn.Linear(in_features=in_features_lin, out_features=_out_features_lin)),
                 (name+'_2_linear', nn.Linear(in_features=_out_features_lin, out_features=out_features_lin)),
                 (name+'_3_linear',  nn.Unflatten(1,(chan, 10, 32, 32))),
                 (name+'relu2',nn.ReLU(inplace=True)),
                 (name+'bn2', nn.BatchNorm2d(out_channels)),
                ])).to(self.device)
        x_lin=lin(xx).to(self.device)
        #print(x_lin.shape)
    
        dec3_t = self.trans3(x_lin).to(self.device)
        concat_skip_3 = torch.cat([_skip_connection_2, dec3_t],axis=1)
        dec3 = self.dec3(concat_skip_3).to(self.device)
        
        #get connection
        dec2_t = self.trans2(dec3).to(self.device)
        concat_skip_2 = torch.cat([_skip_connection_1, dec2_t],axis=1)
        dec2 = self.dec2(concat_skip_2).to(self.device)

        dec1 = self.dec1(dec2).to(self.device)

        _pred=self.conv_final(dec1).to(self.device)
        pred=self.final_softmax(_pred).to(self.device)


        return pred

    @staticmethod
    def _block(in_channels, out_channels, name):
        k = 3
        p = 1
        s = 1
        return nn.Sequential(
            OrderedDict(
                [(name+'_1', nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=k, stride = s, padding=p, bias=False)),
                 (name+'relu1',nn.ReLU(inplace=True)),
                 (name+'bn1', nn.BatchNorm2d(out_channels)),
                 (name+'_2', nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=k, stride = s, padding=p, bias=False)),
                 (name+'relu2',nn.ReLU(inplace=True)),
                 (name+'bn2', nn.BatchNorm2d(out_channels)),
                ]))

    
    
    @staticmethod
    def _trans(out_channels, name):
      k = 2
      p = 0
      s = 2
      return nn.Sequential(
          OrderedDict(
              [(name+'_t', nn.ConvTranspose2d(in_channels=out_channels, out_channels=out_channels, kernel_size=k, stride = s, padding=p))
              ]))