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

### 


### boa
did a specific run adding fhc for abhi

### RNOH
external dataset

### March 24
see the following link for the data statistics
https://unioxfordnexus-my.sharepoint.com/:p:/r/personal/kebl7678_ox_ac_uk/_layouts/15/Doc.aspx?sourcedoc=%7B447DAB8E-0E2B-43A1-B25D-547257BD639C%7D&file=Presentation.pptx&action=edit&mobileredirect=true

### subset is a subset group of 14, 3, 3 to test functionalities