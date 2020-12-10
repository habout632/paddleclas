
### Training

#### train model
~~~python
export CUDA_VISIBLE_DEVICES=0
python -m paddle.distributed.launch \
    --selected_gpus="0" \
    tools/train.py \
        -c ./configs/quick_start/ResNet50_vd_finetune.yaml
~~~

#### eval model
~~~python
python tools/eval_multi_platform.py \
    -c ./configs/eval.yaml \
    -o ARCHITECTURE.name="ResNet50_vd" \
    -o pretrained_model=path_to_pretrained_models
~~~

### export inference model

~~~python
python tools/export_model.py \
    --model=model_name \
    --pretrained_model=pretrained_model_dir \
    --output_path=save_inference_dir
~~~

### Inference
do inference ops

~~~python
py_infer.py
-i=/data/datasets/AFHQ/test/dog/flickr_dog_000572.jpg
-d=../../inference/afhq/
-m=infer.pdmodel
-p=infer.pdparams
--use_gpu=True
~~~


~~~python

infer.py
--i=/data/datasets/AFHQ/test/dog/flickr_dog_000572.jpg
--m=ResNet50_vd
--p=../../output/ResNet50_vd/14/ppcls
--use_gpu=True
~~~


~~~python
predict.py
-i=/data/datasets/AFHQ/test/wild/flickr_wild_000038.jpg
-m=../../inference/afhq/infer.pdmodel
-p=../../inference/afhq/infer.pdparams
--use_gpu=1
--use_tensorrt=False
~~~


~~~python
export_model.py
--m=ResNet50_vd
--p=../output/ResNet50_vd/14/ppcls
--o=../inference/afhq
~~~
##### Note:
m: architechture \
**p: trained model dir with model files name as prefix** \
o: inference model output dir

