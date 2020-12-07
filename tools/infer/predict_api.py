import base64
import json
import traceback

from flask import Flask, request
from flask_cors import CORS
from munch import Munch
from paddle.fluid.core_avx import AnalysisConfig, create_paddle_predictor
import numpy as np
import logging
import cv2
import utils

# Flask
app = Flask(__name__)
CORS(app, resources=r'/*', supports_credentials=False)

# logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

args = {
    "batch_size": 1,
    # "model_file": "../../inference/afhq/infer.pdmodel",
    # "params_file": "../../inference/afhq/infer.pdparams",
    "model_file": "/app/inference/afhq/infer.pdmodel",
    "params_file": "/app/inference/afhq/infer.pdparams",
    "use_gpu": True,
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


# def preprocess(fname, ops):
#     """
#
#     :param fname:
#     :param ops:
#     :return:
#     """
#     data = open(fname, 'rb').read()
#     for op in ops:
#         data = op(data)
#
#     return data


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


@app.route('/car/get/orientation', methods=['POST', 'GET'])
def get_car_orietation():
    # apidoc -i api/ -o apidoc/
    """
       @api {POST} /car/get/orientation 根据图片获取汽车朝向
       @apiName /movie/query/detailprice
       @apiGroup movie_query
       @apiParam {string} image_file 图片文件　base64格式(必填)
       @apiParam {string} cinema_id 马踏飞燕放映电影院ID(必填)
       @apiParam {string} uuid  前端查询请求唯一ID(必填)
       @apiSuccessExample {json} Success-Response:
       HTTP/1.1 200 OK
      {
        "code": 200,
        "data": {
            "uuid":"ecaddfb5-cd51-44ee-a6c1-7da34aa88499",  #前端查询请求唯一ID String
            "cinema_name" : "完美云太郎影城", #电影院名称 String
            "cinema_address" : "朝阳区管庄美廉美超市（云影小镇）一层",#电影院地址 String
            "cinema_phone" : "8888888",#电影院电话 String
            "lat": 12.09801,#影院位置百度纬度 float
            "lng": 112.09871, #影院位置百度精度 float
            "movie_name":"复仇者联盟4",#电影名称  String
            "score":9.5#电影评分 float
        },
        "message": "SUCCESS"
      }

       """
    try:
        datajson = request.get_data()
        # flask_movie_query_logger.info(datajson)
        if datajson:
            dict_info = json.loads(datajson.decode("utf-8"))
            image_str = dict_info.get('image_file')
            #
            starter = image_str.find(',')
            image_str = image_str[starter + 1:]

            img_data = base64.b64decode(image_str)
            nparr = np.fromstring(img_data, np.uint8)
            #
            # image_bytes = image_str.encode()
            # image_data = base64.decodebytes(image_bytes)

            # save image to local file
            # with open("imageToSave.jpg", "wb") as fh:
            #     fh.write(base64.decodebytes(image_data.encode()))

            # from string
            # nparr = np.fromstring(image_data, np.uint32)

            # nparr = np.frombuffer(img_data, dtype=np.int32)
            img = cv2.imdecode(nparr, flags=cv2.IMREAD_COLOR)



            # movie_id = dict_info.get('movie_id')
            # cinema_id = dict_info.get('cinema_id')
            # image_file = "/data/datasets/AFHQ/test/wild/flickr_wild_000038.jpg"

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

            result = {
                "code": 200,
                "data": {
                    "cls": str(cls),
                    "score": str(score),
                    "label": ""

                },
                "message": "check_data[1]"
            }
        else:
            result = {
                "code": 400,
                "data": {},
                "message": "请求参数不能为空!"
            }
        return json.dumps(result)
    except:
        s = traceback.format_exc()
        print(s)
        # flask_movie_query_logger.error(s)
        return json.dumps({'result': 500, 'message': '请求异常'})


if __name__ == "__main__":
    app.run("0.0.0.0", 6666)
