### Table 1 and Table 3
*Scripts*: 

`/deep_fourier_reg/evaluate_oasis.py` for all models, except VoxelMorph, Voxelmorph-Huge and TransMorph;
`baseline_models/transmorph/evaluate_vxm_oasis.py` for VoxelMorph and VoxelMorph-Huge;
`baseline_models/transmorph/evaluate_transmorph_oasis.py` for TransMorph.

*Command line arguments:*
* Fourier-Net (16 channels)

  Full resolution: `--exp_num 54`
  
  Reduced resolution: `--exp_num 70`
* Fourier-Net (32 channels)
  
   Full resolution: `--exp_num 84`
  
   Reduced resolution: `--exp_num 83`

* VoxelMorph (if you have some problems with vxm on reduced resolution, try to change `inshape` variable from full image shape to halved shape)
  
   Full resolution: `--exp_num 13`
  
   Reduced resolution: `--exp_num 27`

* VoxelMorph-Large

  Full resolution: `--exp_num 139`
  
  Reduced resolution: `--exp_num 140`

* VoxelMorph-Huge
  
  Full resolution: `--exp_num 2`
  
  Reduced resolution: `--exp_num 28`

* TransMorph

  Full resolution: `--exp_num 1`
  
* FNO (small)

  Full resolution: `--exp_num 75`
  
  Reduced resolution: `--exp_num 76`
  
* FNO (medium)

  Full resolution: `--exp_num 74`
  
  Reduced resolution: `--exp_num 65`

* FNO (large)

  Full resolution: `--exp_num 71`
  
  Reduced resolution: `--exp_num 69`
  
* FNOReg (small)

  Full resolution: `--exp_num 78`
  
  Reduced resolution: `--exp_num 79`
  
* FNOReg (medium)

  Full resolution: `--exp_num 77`
  
  Reduced resolution: `--exp_num 81`

* FNOReg (large)

  Full resolution: `--exp_num 80`
  
  Reduced resolution: `--exp_num 82`

### Table 2 and Table 4
*Scripts*: 

`/deep_fourier_reg/evaluate_oasis3d.py` for all models, except VoxelMorph, Voxelmorph-Huge and TransMorph;
`baseline_models/transmorph/evaluate_vxm_oasis3d.py` for VoxelMorph and VoxelMorph-Huge;
`baseline_models/transmorph/evaluate_transmorph_oasis3d.py` for TransMorph.

TODO

### Figure 4
* Run evaluation scripts on full resolution for Fourier-Net (32 channels), Vxm-Large, Vxm-Huge, TransMorph, FNO (large), FNOReg (large)
* Move all files `plot_*.png` to folder with script `paper_plot_script.py`
* Run command: `python paper_plot_script.py`
  
