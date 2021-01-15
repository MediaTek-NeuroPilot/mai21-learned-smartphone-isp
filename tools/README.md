## [Optional] Some Useful Tools for Raw-to-RGB Conversion
We also provide some useful tools which might help as follows:

### 1. Calculate MACs and the parameter number

1. install [NeuroPilot](file://PC19012316/Users/mtk19837/Downloads/neuropilot-3.2025.3.tar.gz)  (note: need to update link to make is public)
2. use our provided bash script:
```bash
bash tf_convert_punet-count_gmacs.sh
```

Or some commands like below:
```bash
~/anaconda3/envs/AI/neuropilot/bin/toco \
  --input_file=../models/original/punet_pretrained.pb \
  --output_file=../models/original/punet_pretrained.neuropilot.tflite \
  --input_shape=1,1472,1984,4 \
  --input_array=Placeholder \
  --output_array=output_l0 \
  --allow_nudging_weights_to_use_fast_gemm_kernel
```
Notes: 
* `input_shape` is the for the network input, which is after debayering/demosaicing. If the raw image shape is `(img_h, img_w, 1)`, `input_shape` should be `(img_h/2, img_w/2, 4)`.
* `toco` will generate `output_file`, which can be ignored here.

<br/>

### 2. Convert raw DNG camera files to PUNET's input format

1. install [rawpy](https://pypi.org/project/rawpy/):
```bash
pip install rawpy
```
2. prepare a folder forput all DNG files (e.g. `full_resolution_dng_images/`).
3. use our provided bash script:
```bash
bash dng2png.sh
```

Or some commands like below:
```bash
python dng2png.py ../raw_images/test/full_resolution_dng_images/ ../raw_images/test/fujifilm_full_resolution/
```

<br/>

