import torch 
import torch.nn as nn
from collections import OrderedDict


class unet_meta_lastlayer(nn.Module):
    def __init__(self, bs,img_size,num_meta_feats,in_channels=1, init_features=1, out_channels=1):
        super(unet_meta_lastlayer, self).__init__()
        stride = 2
        pool = 2
        features = int(init_features)
        self.features=init_features
        self.in_channels = in_channels
        self.bs = bs
        self.im_size_h = img_size[0]
        self.im_size_w = img_size[1]
        self.num_meta_feats = num_meta_feats

        self.encoder1 = unet_meta_lastlayer._block(in_channels, features, 'enc1')
        self.pool1 = nn.MaxPool2d(pool, stride=stride)
        self.encoder2 = unet_meta_lastlayer._block(features, features*2, 'enc2')
        self.pool2 = nn.MaxPool2d(pool, stride=stride)
        self.encoder3 = unet_meta_lastlayer._block(features*2, features*4, 'enc3')
        self.pool3 = nn.MaxPool2d(pool, stride=stride)

        self.bottleneck = unet_meta_lastlayer._block(features*4, features*4, 'btl')

        self.trans3 = unet_meta_lastlayer._trans(features*4, 'trans3')
        self.dec3 = unet_meta_lastlayer._block(features*4+features*2,features*2, 'dec3')

        self.trans2 = unet_meta_lastlayer._trans(features*2, 'trans2')
        self.dec2 = unet_meta_lastlayer._block(features*2+features, features, 'dec2')

        self.dec1 = unet_meta_lastlayer._block(features,out_channels, 'dec1')

        self.conv_final = nn.Conv2d(out_channels,out_channels,kernel_size=3, stride=1, padding=1)
        self.final_softmax = nn.Sigmoid()
        self.out_channels = out_channels

    def two_d_softmax(self,x):
        exp_y = torch.exp(x)
        return exp_y / torch.sum(exp_y, dim=(2, 3), keepdim=True)

    def forward(self, x, meta, zeros=False):
        #save skip connection temp for saving
        _skip_connections = []
        
        #encoder
        #print('input size',x.size())
        enc1 = self.encoder1(x.to(torch.float32))
        #print('enc1 size',enc1.size())
        _skip_connection_1 = enc1

        enc1_pool=self.pool2(enc1)

        enc2 = self.encoder2(enc1_pool)
        _skip_connection_2 = enc2
        # print('enc2 size',enc2.size())
        enc2_pool=self.pool3(enc2)
        
        enc3 = self.encoder3(enc2_pool)
        #print('enc3 size',enc3.size())

        #bottleneck
        btl = self.bottleneck(enc3)
        #print('btlneck size',btl.size())

        #decoder
        #print('skip size 1',_skip_connection_1.size())
        #print('skip size 2',_skip_connection_2.size())

        dec3_t = self.trans3(btl)
        # #print('transConv3',dec3_t.size())
        concat_skip_3 = torch.cat([_skip_connection_2, dec3_t],axis=1)
        # #print('concat',concat_skip_3.size())
        dec3 = self.dec3(concat_skip_3)
        #print('dec3', dec3.size())
        
        #get connection
        #print('skip size',_skip_connection_1 .size())
        dec2_t = self.trans2(dec3)
        #print('transConv2',dec2_t.size())

        concat_skip_2 = torch.cat([_skip_connection_1, dec2_t],axis=1)
        #print('decoder2 size w/skip',concat_skip_2.size())
        dec2 = self.dec2(concat_skip_2)

        dec1 = self.dec1(dec2)
        #print('decoder size',dec1.size())

        _pred=self.conv_final(dec1)
        #print('final conv size',_pred.size())
        
        xx=_pred
        bs = self.bs
        #print("pred; ",xx.shape)
        feat = (bs, self.out_channels, self.im_size_h,self.im_size_w)
        #print(feat)
        feats = bs*self.im_size_h*self.im_size_w
        xx=xx.view(-1,feat[1])
        #print('xx before meta: ',xx.size())
        
        #print(meta.shape)
        meta_shape = meta.shape[1]
        
        meta_flat=meta.view(-1,meta_shape)
        meta_flat=meta_flat.repeat(1,self.out_channels)

        if zeros==True:
            meta_flat = torch.zeros(meta_flat.shape)
        
        xx=torch.cat((xx,meta_flat),dim=0)
        #print(xx.shape)
        #flatten to linear layer
        xx=torch.flatten(xx.view(xx.size(1), -1))
        #print('xx:', xx.shape)
        
        in_features_lin = list(xx.shape)[0] #feats + self.num_meta_feats*self.bs
        _out_features_lin = self.num_meta_feats*self.bs #list(xx.shape)[0] #
        out_features_lin = feats * self.out_channels

        #print('in_features_lin:', in_features_lin)
        #print('_out_feats:', _out_features_lin)
        #print('out_feats:', out_features_lin)
        
        name = "meta"
        lin = nn.Sequential(OrderedDict(
                [(name+'_1_linear', nn.Linear(in_features=in_features_lin, out_features=_out_features_lin)),
                 (name+'_2_linear', nn.Linear(in_features=_out_features_lin, out_features=out_features_lin)),
                ]))
        x_lin=lin(xx)
        #print('out:',x_lin.shape)
    
        un_flat=nn.Sequential(nn.Unflatten(0, (bs,self.out_channels,self.im_size_h,self.im_size_w)))

        x_lin=un_flat(x_lin)
        #print('un_flat',x_lin.shape)

        #x_lin = x_lin.transpose(0,1)
        #
        #print('new_shape:',x_lin.shape)

        #pred=self.final_softmax(x_lin)
        pred=self.two_d_softmax(x_lin)
        #print('final size',pred.size())

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

