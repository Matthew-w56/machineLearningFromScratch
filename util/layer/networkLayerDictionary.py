
from util.layer import denseLayer
from util.layer import batchNormLayer
from util.layer import batchNormDenseLayer


classes = {
    "DenseLayer": denseLayer,
    "BatchNormLayer": batchNormLayer,
    "BatchNormDenseLayer": batchNormDenseLayer
}


def get_layer_from_json(json_obj):
    layer_class = classes[json_obj['class_name']]
    return layer_class.build_from_json(json_obj)
