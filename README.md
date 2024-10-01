# FNOReg: Resolution-Robust Medical Image Registration Method Based on Fourier Neural Operator

This is an official implementation of paper FNOReg: Resolution-Robust Medical Image Registration Method Based on Fourier Neural Operator

## Installation
```
git clone https://github.com/anac0der/fnoreg.git
cd fnoreg
python3 -m venv .
. bin/activate
pip install -r requirements.txt
```

## Dataset downloading 

`HERE SMTH ABOUT DATASET LOADING`

## Launching experiments and reproducing the results

After installation of all required dependencies and downloading the data, you need to download model checkpoints in order to reproduce results from our paper. It can be done by running 
`download_ckpt.sh` script:
```
./download_ckpt.sh
```

To get further instructions about reproduction of metrics values in our paper, you can read `instructions_to_reproduce.txt`.

**Brief explanation of main commands** (if you want to launch your own experiments):

***Training scripts***: `train_*.py`, in folder `/deep_fourier_reg` for FNO-based models, VoxelMorph-Large and Fourier-Net and in folder `/baseline_models/transmorph` for VoxelMorph, VoxelMorph-Huge and TransMorph.

To launch the model training, firstly fill in the corresponding config file (with regexp `params_*.json`) and then run the following command:
```
python3 train_*.py --gpu_num gpu_num --size size
```
Here `gpu_num` is number of GPU device in your system and `size` is the size of smallest dimension of input data shape (160 for full resolution and 80 for halved resolution).

***Evaluation scripts***: files with pattern `evaluate_*.py`.
After running of the experiment you can see its number in console output. To evaluate the experiment with number N, run the following command:
```
python3 evaluate_*.py --gpu_num gpu_num --exp_num N --ckpt_epoch ckpt_epoch
```

Here:

* `gpu_num` is the same as in the training scripts;
* `ckpt_epoch` is the number of epoch from which you want to download the checkpoint. This argument should be omitted if you want to evaluate the final model (model with weights after all epochs).


