## [Optional] Some Useful Tools for Raw-to-RGB Conversion
We also provide some useful tools which might help as follows:

### 1. Calculate MACs and the parameter number

1. install [millify](https://pypi.org/project/millify/):
```bash
pip install millify
```
2. use the commands like below:
```bash
python count_stats.py --pb_fpath ../moddels/original/punet_pretrained.pb --MAC
```

<br/>

### 2. Convert raw DNG camera files to PUNET's input format

1. install [rawpy](https://pypi.org/project/rawpy/):
```bash
pip install rawpy
```
2. prepare a folder forput all DNG files (e.g. `full_resolution_dng_images/`).
3. use some commands like below:
```bash
python dng2png.py ../raw_images/test/full_resolution_dng_images/ ../raw_images/test/fujifilm_full_resolution/
```
* 1st argument: input directory with DNG files
* 2nd argument: output directory

<br/>

### 3. Modify file name suffix

Use some commands like below:
```bash
python edit_suffix.py ../results/punet_MAI/ ../results/punet_MAI_upload/ "-punet_pretrained" ""
```
* 1st argument: input directory with files waiting for editing file name suffix
* 2nd argument: output directory
* 3rd argument: the suffix to be removed
* 4th argument: the suffix to be added

<br/>

### 4. Example Jupyter Notebook (Unofficial)

Unofficial Jupyter Notebook [Learned_Smartphone_ISP_Baseline.ipynb](Learned_Smartphone_ISP_Baseline.ipynb) provided by [sayakpaul](https://github.com/sayakpaul). Feel free to check it!

