from models import decisionTree
from models import linearRegression
from models import logisticRegression
from models.networks import layeredNetwork
import json

classes = {
    "LinearRegression": linearRegression,
    "LogisticRegression": logisticRegression,
    "DecisionTree": decisionTree,
    "LayeredNetwork": layeredNetwork
}


def get_existing_model_from_json(filename):
    file = open(filename, )
    json_obj = json.load(file)
    file.close()
    return classes[json_obj["class_name"]].build_from_json(json_obj)


# def build_new_linear_regression()
