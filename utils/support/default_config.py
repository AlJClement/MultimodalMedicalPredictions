from yacs.config import CfgNode as CN

C = CN()

C.DATASET = CN()

C.INPUT_PATHS = CN()
C.INPUT_PATHS.DATASET_NAME = ""
C.INPUT_PATHS.PARTITION = ""
C.INPUT_PATHS.IMAGES = ""
C.INPUT_PATHS.LABELS = ""
C.INPUT_PATHS.META_PATH = ""
C.INPUT_PATHS.ID_COL = ""
C.INPUT_PATHS.META_COLS = []


C.MODEL = CN()
C.MODEL.NAME = ""
C.MODEL.IN_CHANNELS = 0
C.MODEL.OUT_CHANNELS = 0
C.MODEL.INIT_FEATURES = 0
C.MODEL.META_FEATURES = []
C.MODEL.DEVICE = 'cpu' 

#for segmentation modelsOD
C.MODEL.BATCH_NORM_DECODER=True
C.MODEL.ENCODER_NAME = "resnet34"
C.MODEL.ENCODER_WEIGHTS = "imagenet"
C.MODEL.DECODER_CHANNELS = []
C.MODEL.IN_CHANNELS = 1


C.DATASET = CN()
C.DATASET.COMBINE_REVIEWERS=False
C.DATASET.IMAGE_EXT=".png"
C.DATASET.ANNOTATION_TYPE= "LANDMARKS" 
C.DATASET.NUM_LANDMARKS = 5
C.DATASET.PIXEL_SIZE = [1,1]
C.DATASET.CACHED_IMAGE_SIZE= [512,256]
C.DATASET.SIGMA = 0 #if sigma is NONE no guassian applied
C.DATASET.FLIP_AXIS = False

C.DATASET.AUGMENTATION = CN()
C.DATASET.AUGMENTATION.ROTATION_FACTOR = 0
C.DATASET.AUGMENTATION.INTENSITY_FACTOR = 0.0
C.DATASET.AUGMENTATION.SF = 0.0
C.DATASET.AUGMENTATION.TRANSLATION_X = 0.0
C.DATASET.AUGMENTATION.TRANSLATION_Y = 0.0
C.DATASET.AUGMENTATION.ELASTIC_STRENGTH = 0
C.DATASET.AUGMENTATION.ELASTIC_SMOOTHNESS = 0

C.OUTPUT_PATH = CN()
C.OUTPUT_PATH = "/output"

C.TRAIN = CN()
C.TRAIN.BATCH_SIZE = 6
C.TRAIN.LR = 0.001
C.TRAIN.EPOCHS = 10
C.TRAIN.LOSS = ''
C.TRAIN.L2_REG = True

C.TEST = CN()
C.TEST.NETWORK = ''
C.TEST.COMPARISON_METRICS = []
C.TEST.SDR_THRESHOLD = []

def get_cfg_defaults():
    """Get a yacs CfgNode object with default values above."""
    return C.clone()