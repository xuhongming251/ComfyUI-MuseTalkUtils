
from .musetalk_preprocess import *
from .musetalk_postprocess import *
from .musetalk_train_preprocess import *
from .musetalk_train import *


NODE_CLASS_MAPPINGS = {

    # "MuseTalkUncropMask":MuseTalkUncropMask,

    "MuseTalkPreprocess": MuseTalkPreprocess,
    "MuseTalkPostprocess": MuseTalkPostprocess,

    "MuseTalkTrainPreprocess": MuseTalkTrainPreprocess,
    "MuseTalkTrain": MuseTalkTrain,

}


NODE_DISPLAY_NAME_MAPPINGS = {

    # "MuseTalkUncropMask": "MuseTalkUncropMask",

    "MuseTalkPreprocess": "MuseTalkPreprocess",
    "MuseTalkPostprocess": "MuseTalkPostprocess",

    "MuseTalkTrainPreprocess": "MuseTalkTrainPreprocess",
    "MuseTalkTrain": "MuseTalkTrain",

}