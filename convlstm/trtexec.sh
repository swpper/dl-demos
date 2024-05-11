#!/bin/bash
# TensorRT推理脚本

ENGINE_PATH=engine.trt

ONNX_MODEL_PATH=model.onnx
# aws s3 cp s3://staging-lightning/tensorrt/624094514d8643278b8172e5dab703b4/area_model_624094514d8643278b8172e5dab703b4.onnx $ONNX_MODEL_PATH

# ONNX_MODEL_PATH=/tmp/tensorrt_engine/area_model_624094514d8643278b8172e5dab703b4_optimized.onnx
# aws s3 cp s3://staging-lightning/tensorrt/624094514d8643278b8172e5dab703b4/area_model_624094514d8643278b8172e5dab703b4_optimized.onnx $ONNX_MODEL_PATH

trtexec \
    --onnx=$ONNX_MODEL_PATH \
    --saveEngine=$ENGINE_PATH \
    --minShapes=encoder_input:1x5x1x32x32 \
    --optShapes=encoder_input:16x5x1x32x32 \
    --maxShapes=encoder_input:32x5x1x32x32 \
    --workspace=1024 \
    --fp16 \
    --best \
    --verbose \
    > ./trtexec.log 2>&1

# aws s3 cp $ENGINE_PATH s3://staging-lightning/tensorrt/624094514d8643278b8172e5dab703b4/area_model.trt

echo "TensorRT engine has been saved to S3."