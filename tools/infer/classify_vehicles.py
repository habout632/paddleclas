import base64
import json
import os
import traceback

from flask import Flask, request
from flask_cors import CORS
from munch import Munch
from paddle.fluid.core_avx import AnalysisConfig, create_paddle_predictor
import numpy as np
import logging
import cv2
import utils

# logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

args = {
    "batch_size": 1,
    "model_file": "../../inference/vehicles/infer.pdmodel",
    "params_file": "../../inference/vehicles/infer.pdparams",
    # "model_file": "/app/inference/afhq/infer.pdmodel",
    # "params_file": "/app/inference/afhq/infer.pdparams",
    "use_gpu": False,
    "use_tensorrt": True,
    "ir_optim": True,
    "gpu_mem": 8000,
    "enable_benchmark": False,
    "use_fp16": False,
    "ir_optim": True
}

args = Munch(args)


def create_predictor(args):
    config = AnalysisConfig(args.model_file, args.params_file)

    if args.use_gpu:
        config.enable_use_gpu(args.gpu_mem, 0)
    else:
        config.disable_gpu()

    config.disable_glog_info()
    config.switch_ir_optim(args.ir_optim)  # default true
    if args.use_tensorrt:
        config.enable_tensorrt_engine(
            precision_mode=AnalysisConfig.Precision.Half
            if args.use_fp16 else AnalysisConfig.Precision.Float32,
            max_batch_size=args.batch_size)

    config.enable_memory_optim()
    # use zero copy
    config.switch_use_feed_fetch_ops(False)
    predictor = create_paddle_predictor(config)

    return predictor


def create_operators():
    size = 224
    img_mean = [0.485, 0.456, 0.406]
    img_std = [0.229, 0.224, 0.225]
    img_scale = 1.0 / 255.0

    decode_op = utils.DecodeImage()
    resize_op = utils.ResizeImage(resize_short=256)
    crop_op = utils.CropImage(size=(size, size))
    normalize_op = utils.NormalizeImage(
        scale=img_scale, mean=img_mean, std=img_std)
    totensor_op = utils.ToTensor()

    # return [decode_op, resize_op, crop_op, normalize_op, totensor_op]
    return [resize_op, crop_op, normalize_op, totensor_op]


def preprocess(img, ops):
    """

    :param fname:
    :param ops:
    :return:
    """
    # data = open(fname, 'rb').read()
    for op in ops:
        data = op(img)

    return data


operators = create_operators()
predictor = create_predictor(args)

input_names = predictor.get_input_names()
input_tensor = predictor.get_input_tensor(input_names[0])

output_names = predictor.get_output_names()
output_tensor = predictor.get_output_tensor(output_names[0])


def classify_vehicles(img_path):
    """
    classfiy_vehicles
    """
    img = cv2.imread(img_path)
    inputs = preprocess(img, operators)
    inputs = inputs.astype("float32")
    inputs = np.expand_dims(
        inputs, axis=0).repeat(
        args.batch_size, axis=0).copy()
    input_tensor.copy_from_cpu(inputs)

    predictor.zero_copy_run()

    output = output_tensor.copy_to_cpu()
    output = output.flatten()
    cls = np.argmax(output)
    score = output[cls]
    logger.info("class: {0}".format(cls))
    logger.info("score: {0}".format(score))


if __name__ == "__main__":
    pwd = "/home/habout632/Datasets/vehicle/test/others/"
    for filename in os.listdir(pwd):
        try:
            img_path = pwd+filename
            print(img_path)
            classify_vehicles(img_path)
            # print(result)
        except Exception as e:
            print(str(e))
