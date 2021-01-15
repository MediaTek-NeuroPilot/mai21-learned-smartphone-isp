#!/bin/bash         

~/anaconda3/envs/AI/neuropilot/bin/toco \
  --input_file=../models/original/punet_pretrained.pb \
  --output_file=../models/original/punet_pretrained.neuropilot.tflite \
  --input_shape=1,1488,1984,4 \
  --input_array=Placeholder \
  --output_array=output_l0 \
  --allow_nudging_weights_to_use_fast_gemm_kernel
