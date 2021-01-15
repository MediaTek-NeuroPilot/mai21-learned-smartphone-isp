#!/bin/bash         

tflite_convert \
  --graph_def_file=models/original/punet_pretrained.pb \
  --output_file=models/original/punet_pretrained.tflite \
  --input_shape=1,1488,1984,4 \
  --input_arrays=Placeholder \
  --output_arrays=output_l0
