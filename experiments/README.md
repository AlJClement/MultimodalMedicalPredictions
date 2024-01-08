# READ ME

This document explains the different experiment files

## Baseline Models

### ddh_190
This is the baseline experiment. Ran with L2 Regularization. Ddh_190_noreg is the same but L2 Reg OFF.
For this only one folder is placed in the annotation path /txt/_g.

### ddh_570_allreviwers
This file leverages all 3 reviewrs (_g,_g10,g10w) by loading them all.
In config.
  COMBINE_REVIEWERS: False #default 

NOTE: this is the same as ddh_190 but the difference is that there are now three folders in the annotation path
/txt/_g, /txt/_g10, /txt/_g10w 

### ddh_190_combinereviewers


## Late Fusion Models
### ddh_meta_latefusion