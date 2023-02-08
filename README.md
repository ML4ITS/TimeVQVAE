# TimeVQVAE
This is an official Github repository for the PyTorch implementation of TimeVQVAE from the paper ["Vector Quantized Time Series Modeling with a Bidirectional Prior Model", AISTATS 2023].

TimeVQVAE is a robust time series generation model that utilizes vector quantization for data compression into the discrete latent space (stage1) and a bidirectional transformer for the prior learning (stage2).

<p align="center">
<img src=".fig/stage1.jpg" alt="" width=100% height=100%>
</p>

<p align="center">
<img src=".fig/stage2.jpg" alt="" width=50% height=50%>
</p>

<p align="center">
<img src=".fig/iterative_decoding_process.jpg" alt="" width=100% height=100%>
</p>

<p align="center">
<img src=".fig/example_of_iterative_decoding.jpg" alt="" width=60% height=60%>
</p>


## Install / Environment setup
The following command creates the conda environment from the `environment.yml`. The installed environment is named `timevqvae`.
```
$ conda env create -f environment.yml
```
You can activate the environment by running
```
$ conda activate timevqvae
```

## Usage

### Configuration
- `configs/config.yaml`: configuration for dataset, data loading, optimizer, and models (_i.e.,_ encoder, decoder, vector-quantizer, and MaskGIT)
- `config/sconfig_cas.yaml`: configuration for running CAS, Classification Accuracy Score (= TSTR, Training on Synthetic and Test on Real).

### Usage
:rocket: The stage 1 and stage 2 training can be performed with the following command: 
```
$ python stage1.py   
```
```
$ python stage2.py   
```
Note that you need to specify a dataset of your interest in `configs/config.yaml`.

:bulb: The training pipeline is as follows:
- Run `stage1.py` and it saves trained encoders, decoders, and vector-quantizers for LF and HF.
- Run `stage2.py` and it saves the prior model (_i.e.,_ bidirectional transformer).
  - `stage2.py` includes an evaluation step which is performed right after the stage 2 training. The evaluation includes a visualization plot of test samples (from a test set) versus generated samples, FID score, and IS (Inception Score).    

:rocket: If you want to run stage 1 and stage 2 at the same time, use the following command. You can specify dataset(s) and a GPU device in the command line for `stages12_all_ucr.py`.
```
$ python stage12_all_ucr.py --dataset_names CBF BME --gpu_device_idx 0
```

:rocket: CAS can be performed with the following command:
```
$ python run_CAS.py  --dataset_names CBF BME --gpu_device_idx 0
```


## Citations
```
@article{...,
  title={...},
  author={...},
  journal={...},
  volume={},
  pages={},
  year={2023},
  publisher={}
}
```
