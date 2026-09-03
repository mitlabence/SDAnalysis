# SDAnalysis
The data required along with this code to reproduce Figures 1-3 and the related supplementary figures of the associated publication is available upon request or accessible via the up-to-date public availability link in the linked [repository](https://github.com/mitlabence/SDAnalysis)
# Installation
See `conda\howto.txt`. Alternatively, use uv:
`uv sync --frozen --no-install-project`
# Guide
## Set up .env file
In the root folder of the sdanalysis repository (same folder as .env-sample), use the .env-sample file to set up the following environmental variables:
*`METADATA_FOLDER=(data folder)\Metadata`
*`DOWNLOADS_FOLDER=(downloads folder)`
*`LOG_FOLDER=(arbitrary folder with write permission)`
*`DATA_FOLDER=(data folder)`
*`OUTPUT_FOLDER=(arbitrary folder with write permission)`
## Recovery analysis 

* For TMEV and optogenetic stimulation (with window) (Fig. 1 and S5) `python recovery_analysis.py --save_results --fpath_tmev_dset data-folder\Recovery_analysis\traces_for_recovery_analysis_tmev_20240109-180400.h5 --fpath_stim_dset data-folder\Locomotion_analysis\Window_stimulation\assembled_traces_window-stim.h5`

## Directionality analysis

* Fig. 1 K-N: `python directionality_analysis.py --save_data --folder data-folder\Directionality_analysis\Used\TMEV`

## Locomotion analysis

* Fig. 2 B-E: `python locomotion_analysis.py --save_data --fpath data-folder\Locomotion_analysis\TMEV\assembled_traces_tmev.h5`
* Fig. 3 G-J: `python locomotion_analysis.py --save_data --fpath data-folder\Locomotion_analysis\Window_stimulation\assembled_traces_window-stim.h5`
* Fig. 3M-P, Fig. S8: run `locomotion_analysis_sle.ipynb`
* Fig. S3: run `locomotion_sz_ssd.ipynb`

## Cell count analysis

* Fig. 3F: `python cell_count_analysis.py --save_data --fpath data-folder\Cell_count_analysis\files_for_analysis_pre_post_stim.json`

## SLE patterns

* Fig. S8: run `sle_plotting.ipynb`

## Wavefront speed

* Fig. S5: `python speed_analysis.py --save_data --folder data-folder\Directionality_analysis\Used\TMEV`