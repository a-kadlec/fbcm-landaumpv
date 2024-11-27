# TBTools
This project has been entirely moved to be a part of the BRIL TBTools project: https://gitlab.cern.ch/bril-phase2/fbcmTBTools/-/tree/master/EffectiveThickness?ref_type=heads
This is page is now defunct, get the code and documentation on the above link instead.


# Get the package
You can clone the repository with:
```
git clone https://github.com/a-kadlec/fbcm-landaumpv.git
```
Alternatively, you can download the files by clicking the green "<> Code" button on the repository page.

## Extra requirements:
These need to be installed in your pyenv.
1) landaupy: https://github.com/SengerM/landaupy 
2) pyyaml: https://pypi.org/project/PyYAML/
3) mplhep: https://mplhep.readthedocs.io/en/latest/


# How it works
This package is for fitting a Landau function over the ToT spectrum, and to calculate the effective thickness of the FBCM sensors. This takes 2 steps:
## Calibration: the x-axis of the histogram data must be converted from ns to fC
Calibration data (tot vs collected charge) is taken at the lab - this data is fit with some function to provide a "conversion" between time and charge for the x-axis of the testbeam tot histogram data. Currently the fits can be:
- a logarithmic fit with 3 parameters: a + b*log(x+c)
- a smoothed cubic spline interplation
- linear interpolation

The selected method will be then applied to the histogram data.

## Landau fitting and effective thickness calculation
The strategy of the code is that it will perform two Landau fits to the histogram data: first an 'initial' fit, and then a 'refined' fit.
The reason is that there is a known inhomogenity issue in the sensors, and points that are far to the right tail of the MPV (Most Probable Value of the Landau) are further away from the fit than they should be. Therefore in the second "refined" fit, we try to ignore these points to increase the accuracy of the MPV value. The initial parameters of the refined fit are the parameter results from the initial fit.

The refined fit is the one that will be shown on the plots, and the ignored points are shown with black x marks.

The uncertainty of the MPV that comes from the number of used data points is estimated the following way: two more fits are performed, one where we ignore an additional point, and another where we include one of the ignored points, and then compare how much the MPV value shifts from these changes.

The effective thickness is calculated from the MPV, and is plotted against the threshold values for each channel.


# Board configuration file setup
To run the code, first you need to edit and set up the board configuration yaml file according to the parameters of the currently analyzed board.
<details>
<summary>Details of the config keys</summary>
  
* board_number: (integer) the board number of the currently analyzed board
* color_palette: (str) Specify the used color palatte. Can be "cms" for the 10-color M. Petroff scheme, "cms6" for the 6-color M.Petroff scheme, or "custom" for the legacy custom colors.
* output_dir: (string) the directory where the output plots and files should be created. If the path does not already exist, the code will attempt to create it
* measurement: (string) a descriptive name for the measurement that will be concated to the board number in file names. Should reflect the conditions of the used calibration, for example "CBm14_WS_IRRAD_GRD_20"

* calibration_data: (string) the path to the npz pickle file of the lab calibration. If the board has two ASICs, then put the calibration file for the RIGHT asic here.
* calibration_data_left: (bool/string) If the board has only one ASIC, then thus must be False. If the board has two ASICs, then put the path to the calibration file for the LEFT asic here.

* histogram_data_dir: (string) The directory where the histogram data from the testbeam is stored. This should be "/data/testbeam2/histogram_data"
* histogram_data_files: (string/int/list) The name of the histogram data file, or a list of files (the contents of which will be summed). The full file name can be specified as a string (f.e. "histogram_data_run_538_2024-07-26_07-27-05.dat"), or for convenience, it is enough to specify only the run number (f.e. 538). This is convenient if you want to use a list of files, for example from run 512 to 516, then this can be [512,513,514,515,516].
* RC: (integer) The RC setting that was used while taking the histogram data
* threshold_per_channel: (list[6]) The threshold values for each channel that was used while taking the histogram data
* expected_thickness_per_channel: (list[6]) The expected values for the effective thickness for each channel respectively - the deviation of the results w.r.t. these values will be calculated.
* skip_channels: (list) If a channel number is included in this list, it will be skipped from the analysis. Useful to ignore known noisy channels. Empty list ([]) means none of the channels are skipped, all are analyzed.

* tested_calibration_types: (string): The types of calibration to test and draw on the calibration data. This can be "log" for a logarithmic fit, "spline" for interpolation with a smoothed cubic spline, or "linear" for linear interpolation. It can also be a combination of any of these, for example "log, linear" to test both log fit and linear interpolation.
* selected_calibration_type: (string) The calibration to select for usage on the histogram data. This cannot be any sort of list! One type of calibration must be selected.
* calib_bad_point_errorlimit_ns: (float) A threshold for the sigma of the measurement points in the calibration data: if a point has higher sigma than this, it is ignored from the calibration fit.
* channels_on_calib_summary_plots: (2D list) This setting specifies which channels are shown on which summary plot (where the calibration fits for different channels are compared on the same plot). The first index denotes how many plots to make, and the second index contains which channels are on the plot. For example: [[0,3],[2,4,5]] will make two plots, the first one will show channels 0 and 3, and the second one will show channels 2, 4, and 5. Any number of plots can be made with any number of channels.
* calib_summary_plots_xticks_density: (float) The density of the xticks marks on the x axis for all calibration summary plots. Can be decreased for a cleaner plot, or can be increased to help see differences when comparing nearly identical channels.
* calib_summary_plots_draw_grid (bool): Whether to draw a grid on the calibration summary plots.

* cutoff_ns: (float) This is an upper cut on the x-axis of the raw ToT plot in nanoseconds: any data that belongs to a time bin larger than this value will not be plotted.
* cutoff_fC: (float) This is an upper cut on the x-axis of the landau plot in fC (after calibration has been applied to the x-axis): similarly, any data that belongs to a higher fC bin than this value is not plotted, and is not considered in the initial Landau fit.
* cutoff_ref: (float) This is a cut for the refined Landau fit in percent (%) values. After the initial fit, each point to the right of the MPV is tested against expected value from the initial fit, and if the deviation in percent is larger than this cutoff value, the point will be ignored from the refined fit.
* cutoff_lower: (list[6]) This adds an option to study and decrease the effect of too many near-zero points on the left tail for the refined fit. The numbers in the list specify for each channel, how many points to ignore from the left of the histogram data. For example, [3,0,0,0,0,0] means that only for channel 0, the three lowest points on the x-axis are ignored from the refined fit.

* mpv_display_precision: (int) number of decimals for the MPV to show on the plots
* MPV_summary_plot_separate_stat_syst_errorbars: (bool) If set to True, each point on the MPV vs threshold summary plot will have two errorbars: a colored one denoting the stat, and a black one denoting the stat+syst error. If set to False, only the total stat+syst errorbar is shown in color.

* log_p0: (list[3]) Initial parameters for fitting the logarithm calibration: a + b*log(x+c) You may need to adjust especially the first parameter when changing boards.  
* landau_p0: (list[3]) Initial parameters for the initial Landau fit: normalization factor, MPV, sigma. You may need to adjust the normalization factor when changing boards.
* spline_smoothing_param: (int) smoothing parameter for the cubic spline

</details>

# Running the code
To run the whole code (do the calibration, and then fit landaus and calculate the effective thickness), just run:
```
python3 landaumvp.py <board_config_file_name>.yaml
```
This will produce all plots (calibration, tot, landau fits), a summary plot of MPV and effective thickness vs. threshold for each channel, and will save the results to an output file.


If you want to produce just the calibration plots (for debugging the fits for example), you can do that by running:
```
python3 calib.py <board_config_file_name>.yaml
```

Note: During the calibration fitting, RuntimeWarnings may pop up if the log function is used, because the range of parameters is not constrained at the moment. These warnings have been suppressed by default. However, if there is something wrong with the result of your fits, you can enable all warnings by running the scripts with the -w option for more debugging info.
```
python3 calib.py <board_config_file_name>.yaml -w
```

