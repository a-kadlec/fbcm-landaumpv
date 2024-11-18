import os
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit, OptimizeWarning
import math
import warnings

bShowAllWarnings = False
def set_warnings(b):
    global bShowAllWarnings
    bShowAllWarnings = b
    if not bShowAllWarnings:
        print("Info: Some RuntimeWarnings have been suppressed for the fitting function!")

def get_voltage(dac_map, code):
    for i in dac_map:
        if code == i['code']:
            return i['voltage']
    raise ValueError("Missing code: %s in the dac map." % code)

def tot_func(x, arg0,arg1,arg2): # *args):
    '''
    eq_border = 4
    eq1 = x < eq_border
    eq2_b = args[0] + args[1]*np.sqrt(eq_border)  + args[2]*eq_border

    res = eq2_b + args[3]*(x-eq_border)
    res[eq1] = args[0] + args[1]*np.sqrt(x[eq1])  + args[2]*x[eq1] #-args[2]
    return res
    '''
    #return arg0 + arg1*np.log(x-arg2) + arg3* (np.log(x-arg4))**2

    #if not all([x >= 0 for x in x+arg2]):
    #    return np.array([np.inf]*len(x))

    if not bShowAllWarnings:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return arg0 + arg1*np.log(x+arg2)
    else:
        return arg0 + arg1*np.log(x+arg2)

    

def invert_tot_func(x, a,b,c):
    #return -np.exp(-a/b)*(d*np.exp(a/b) - np.exp(x/b)) / c
    return np.exp((x-a)/b) - c

def nd_dict(array):
    if type(array) == np.ndarray and not array.shape:
        return array.reshape(1)[0]
    else:
        return array
    
def process_npz(npz):
    if type(npz) != dict: # dhould compare == NpzFile:
        npz = dict(npz) 
    for k,v in npz.items():
        npz[k] = nd_dict(v)
    return npz



def get_ASIC_calibration(data, sample, config_file):
    matplotlib.rc('font', size=16)
    
    # check what calibrations to calculate
    bCalibLog = False
    bCalibSpline = False
    bCalibLinear = False
    if isinstance(config_file['tested_calibration_types'], str):
        bCalibLog = "log" in config_file['tested_calibration_types']
        bCalibSpline = "spline" in config_file['tested_calibration_types']
        bCalibLinear = "linear" in config_file['tested_calibration_types']
    elif isinstance(config_file['tested_calibration_types'], list):
        for string in config_file['tested_calibration_types']:
            if "log" in string:
                bCalibLog = True
            if "spline" in string:
                bCalibSpline = True
            if "linear" in string:
                bCalibLinear = True

    # sanity check
    if not bCalibSpline and not bCalibLog and not bCalibLinear:
        raise Exception("Error: No calibration type specified, check your config file.")
    if config_file['selected_calibration_type'] == "log" and not bCalibLog:
        raise Exception("Error: You're trying to use log calibration for the histogram data, but did not specify log as an option for the calibration data. Check your config file.")
    if config_file['selected_calibration_type'] == "spline" and not bCalibSpline:
        raise Exception("Error: You're trying to use spline calibration for the histogram data, but did not specify spline as an option for the calibration data. Check your config file.")
    if config_file['selected_calibration_type'] == "linear" and not bCalibLinear:
        raise Exception("Error: You're trying to use linear interpolation calibration for the histogram data, but did not specify linear interpolation as an option for the calibration data. Check your config file.")



    results = data['toa_tot']['results']
    test_conditions = data['toa_tot']['test_conditions']

    twalk = data['toa_tot']['results']['twalk']


    dac_8bit_maps = data['dacs_8bit']['results']['calibration']
    # channel_th_offset = dac_8bit_chars['channel_th_offset']

    baseline = data['noise_occ']['results']

    step = test_conditions['step']

    print("Calibration fitting for "+sample)
    print("Tested thresholds: ",test_conditions['th_list_fC'])
    print("Tested RCs: ",test_conditions['rccal'])


    output = {}
    output['fitparams'] = []
    output['spline'] = []
    output['linear'] = []
    output['RC'] = []
    output['th'] = []
    output['channel'] = []

    try:
        testboard_channels = test_conditions["testboard_channels"]
        asic_channels = test_conditions["asic_channels"]
    except KeyError:
        testboard_channels = test_conditions["channels"]
        asic_channels = test_conditions["channels"]

    tot_summary = []
    tot_summary_x = []
    tot_std_summary = []
    tot_summary_text = []
    tot_summary_params = []
    tot_summary_fitchcalval = []
    tot_summary_linear = []
    tot_summary_spline = []

    #for testboard_channel in test_conditions['channels']:
    for asic_channel, testboard_channel in zip(asic_channels,testboard_channels):

        if testboard_channel in config_file['skip_channels']:
            print(f"Config file info: skipping channel {testboard_channel} from calibration")
            continue

        toa =     results['toa'][testboard_channel].copy()
        tot =     results['tot'][testboard_channel].copy()


        bUseSigma = False
        try:
            tot_std = results['tot_std'][testboard_channel].copy()
            bUseSigma = True
        except:
            print("Warning: no tot_std key found in results! You are running on old data! (And fits may fail)")

        tw = twalk[testboard_channel].copy()
        # print(tw.shape)

        # chcalval = np.zeros(shape=(256,))
        # for i in np.arange(0,256,step):
        #     chcalval[i] = get_voltage(dac_8bit_maps['CHCALVAL']['node']['CHCALVAL']['cal_char'], i)
        chcalval = dac_8bit_maps['CHCALVAL']['node']['CHCALVAL']['cal_char']


        # err = tot > 1e10
        # toa[err] = 0
        # tot[err] = 0
        for th_level_fC in range(len(test_conditions['th_list_fC'])):
            for r in range(len(test_conditions['rccal'])):
                if test_conditions['rccal'][r] == 0: continue

                try: # sinve v13 tests:
                    if (test_conditions['th_list_fC'][th_level_fC] not in test_conditions['th_list_per_rccal'][test_conditions['rccal'][r]]):
                        continue
                except:
                    pass


                
                output['RC'].append(test_conditions['rccal'][r])
                output['th'].append(test_conditions['th_list_fC'][th_level_fC])
                output['channel'].append(testboard_channel)

                fig = plt.figure(2)
                ax = fig.add_axes([0,0,1,1])
                ax.set_title("FBCM", loc="left")
                ax.set_title('CH: %s, RC=%x, th=%.2f fC' % (testboard_channel, test_conditions['rccal'][r], test_conditions['th_list_fC'][th_level_fC]), loc="right")
                ax.set_ylabel('Time [ns]')
                ax.set_xlabel('Input charge [fC]')
                # start = np.argmin(err[r,0:256:step])*step # find first False what means first valid value
                if 'start' in test_conditions:
                    start = test_conditions['start']
                    if not isinstance(start, int) and not isinstance(start, np.int64):  
                        start = int(test_conditions['start'][testboard_channel,r,th_level_fC])
                elif 'start' in results:
                    start = results['start']
                    if not isinstance(start, int) and not isinstance(start, np.int64):
                        start = int(results['start'][testboard_channel,r,th_level_fC])
                else:
                    start = 0
                end = 255

                tw[r,th_level_fC,start:end:step] =  tw[r,th_level_fC,start:end:step] - np.min(tw[r,th_level_fC,start:end:step])


                xvalues = chcalval[start:end:step]*1e15
                myTot = tot[r,th_level_fC,start:end:step]*1e9

                th_level_fC_val = test_conditions['th_list_fC'][th_level_fC]
                #print(th_level_fC_val)
                mychcalval = np.concatenate(([th_level_fC_val*1e-15],chcalval[start:end:step]))


                if bUseSigma:
                    sigma = tot_std[r,th_level_fC,start:end:step]*1e9

                    for i_s in range(len(sigma)):
                        if sigma[i_s] == 0:
                            sigma[i_s] = 0.0001
                            print("WARNING: An std value of 0 was provided in tot_std.")
                else:
                    sigma = np.array( [1] * len(myTot) )
                


                xvalues_filtered = []
                myTot_filtered = []
                sigma_filtered = []

                xvalues_off = []
                myTot_off = []
                sigma_off = []

                for i_tot in range(len(myTot)):
                    if myTot[i_tot] > 0 and abs(sigma[i_tot]) <= config_file['calib_bad_point_errorlimit_ns']:
                        xvalues_filtered.append(xvalues[i_tot])
                        myTot_filtered.append(myTot[i_tot])
                        sigma_filtered.append(sigma[i_tot])
                    else:
                        xvalues_off.append(xvalues[i_tot])
                        myTot_off.append(myTot[i_tot])
                        sigma_off.append(sigma[i_tot])
                

                if len(myTot_filtered) == 0:
                    print("No valid data points left after filtering zeros and outliers in ToT values.")
                    continue
                    
                bMatchingSetupForChannel = test_conditions['th_list_fC'][th_level_fC] == config_file['threshold_per_channel'][testboard_channel] and test_conditions['rccal'][r] == config_file['RC']


                if bMatchingSetupForChannel:
                    tot_summary.append( myTot_filtered.copy())
                    tot_summary_x.append( xvalues_filtered.copy())
                    tot_std_summary.append(sigma_filtered.copy())
                    tot_summary_text.append({'ch': testboard_channel, 'th': config_file['threshold_per_channel'][testboard_channel]}  )


                if bCalibLog:
                    try:
                        popt, pcov = curve_fit(tot_func, xvalues_filtered, myTot_filtered, p0=config_file['log_p0'], sigma=sigma_filtered, absolute_sigma=True)
                    except RuntimeError as e:
                        print(f"Fit failed for data: xvalues={xvalues_filtered}, myTot={myTot_filtered}, p0={config_file['log_p0']}, RC={test_conditions['rccal'][r]}, channel={testboard_channel}")
                        raise e

                    Nexp = tot_func(xvalues, *popt)
                    Ndif = myTot - Nexp
                    chi2 = np.sum((Ndif/sigma)**2)
                    chi2ondf = chi2 / (len(myTot) - len(popt))

                    output['fitparams'].append(popt.copy())

                    fit_chcalval = np.arange(0.6e-15, mychcalval[-1], (mychcalval[1]-mychcalval[0])/10)*1e15
                    fit_tot = tot_func(fit_chcalval, *popt)

                    if bMatchingSetupForChannel:
                        tot_summary_params.append(popt.copy())
                        tot_summary_fitchcalval.append(fit_chcalval.copy())

                if bCalibLinear:
                    from scipy.interpolate import interp1d

                    interp_result = interp1d(xvalues_filtered, myTot_filtered, fill_value="extrapolate")
                    interp_inverse = interp1d(myTot_filtered, xvalues_filtered , fill_value="extrapolate")

                    if bMatchingSetupForChannel:
                        tot_summary_linear.append(interp_result)

                    output['linear'].append(interp_inverse)

                if bCalibSpline:
                    from scipy.interpolate import CubicSpline
                    splinex = xvalues_filtered
                    spliney = myTot_filtered
                    splinefit = CubicSpline(splinex, spliney)

                    from scipy.interpolate import splrep, BSpline
                    smoothing = config_file['spline_smoothing_param']
                    tck_s = splrep(splinex, spliney, s=smoothing)


                    Nexp = BSpline(*tck_s)(splinex)
                    Ndif = spliney - Nexp
                    chi2_sp = np.sum((Ndif/sigma)**2)
                    chi2ondf_sp = chi2_sp / (len(splinex) - 4)


                    invsplx = spliney.copy()
                    invsply = splinex.copy()

                    # TODO: Implement more efficient linked sorting algorithm than this quick'n'dirty bubble sort
                    bubble = True
                    while(bubble):
                        bubble = False
                        for i in range(1,len(invsplx)):
                            if invsplx[i - 1] > invsplx[i]:
                                temp = invsplx[i]
                                invsplx[i] = invsplx[i - 1]
                                invsplx[i - 1] = temp
                                bubble = True

                    tck_s_inv = splrep(invsplx, invsply, s=smoothing)


                    if bMatchingSetupForChannel:
                        tot_summary_spline.append(tck_s)

                    output['spline'].append(tck_s_inv)
                #print(chcalval[start]*1e15)
                #ax.plot(chcalval[start:end:step]*1e15, toa[r,start:end:step]*1e9, 'sr', label='ToA')

                ax.plot(xvalues_filtered, myTot_filtered, 'sb', label='ToT')
                if bUseSigma:
                    ax.errorbar(xvalues_filtered, myTot_filtered, yerr=sigma_filtered, capsize=3, fmt=".", ecolor = "black")

                if xvalues_off:
                    ax.errorbar(xvalues_off, myTot_off, yerr=sigma_off, capsize=3, fmt="x", ecolor = "black", color="black")
                
                ax.plot(xvalues, tw[r,th_level_fC,start:end:step]*1e9, 'or', label='TW')

                goodlim_x = ax.get_xlim()
                goodlim_y = ax.get_ylim()

                #params = []
                #for i in range(len(pfit[0])):
                #    params.append("%.2f ± %.2f" % (pfit[0][i], perr[i])   )

                if bCalibLog:
                    #params = ["%.2f" % p for p in popt]
                    ax.plot(fit_chcalval, fit_tot, 'g', label=r"Log fit, $\chi^{2}$/dof="+"{:.1f}".format(chi2ondf))    
                    #'Log fit %s' %  params +r      doesn't make much sense to print params (doesn't mean anything without errors, but also not printing errors due to space on plot...)

                x_range = np.linspace(goodlim_x[0], goodlim_x[1],1000)
                if bCalibSpline:
                    ax.plot(x_range, BSpline(*tck_s)(x_range), 'purple', label=f'Cubic spline (smoothing={smoothing})'+r", $\chi^{2}$/dof="+"{:.1f}".format(chi2ondf_sp))
                if bCalibLinear:
                    ax.plot(x_range, interp_result(x_range), 'darkorange', label="Linear interpolation")

                ax.set_xlim(goodlim_x)
                ax.set_ylim(goodlim_y)

                #fig.legend(loc=[0.14,0.82], prop={'size': 10})
                ax.legend(loc='best', prop={'size': 10})  # bbox_to_anchor=(0, 0, 0.9, 0.9)
                # plt.yscale('log')
                fig.savefig(config_file['output_dir']+'/calib_plots/tot_tw_sample_%s_channel_%d_rccal_%d_th_%s.jpg' % (sample, testboard_channel, test_conditions['rccal'][r],  test_conditions['th_list_fC'][th_level_fC]), format='jpg', bbox_inches='tight')
                plt.close()


    tot_summary = {
        "tot_summary": tot_summary,
        "tot_std_summary": tot_std_summary,
        "tot_summary_x": tot_summary_x,
        "tot_summary_params": tot_summary_params,
        "tot_summary_text": tot_summary_text,
        "fit_chcalval": tot_summary_fitchcalval,
        "tot_summary_linear": tot_summary_linear,
        "tot_summary_spline": tot_summary_spline,
    }
    return output, tot_summary




def get_board_calibration(config_file):
    matplotlib.rc('font', size=16)
    
    
    samplenum = config_file['board_number']
    meas = "_"+config_file['measurement']
    sample = str(samplenum) + meas

    if config_file['calibration_data_left']:
        sample = sample + "_R"

    data = process_npz(dict(np.load(config_file['calibration_data'], allow_pickle=True, encoding="ASCII")))

    output, tot_summary = get_ASIC_calibration(data, sample, config_file)

    if config_file['calibration_data_left']:
        data_left = process_npz(dict(np.load(config_file['calibration_data_left'], allow_pickle=True, encoding="ASCII")))
        sample_left = str(samplenum) + meas + "_L"
        output_left, tot_summary_left = get_ASIC_calibration(data_left,sample_left, config_file)

        # ASICS use channels 0,2,5, Right:0,1,2, Left:3,4,5
        # but this is no longer necessary since get_ASIC_calibration now loops on testboard_channels instead of asic_channels
        '''
        for i in range(len(output['channel'])):
            if output['channel'][i] == 2:
                output['channel'][i] = 1
            elif output['channel'][i] == 5:
                output['channel'][i] = 2
        for i in range(len(output_left['channel'])):
            if output_left['channel'][i] == 0:
                output_left['channel'][i] = 3
            elif output_left['channel'][i] == 2:
                output_left['channel'][i] = 4
        '''


        for key in output.keys():
            output[key].extend(output_left[key])

        for key in tot_summary.keys():
            tot_summary[key].extend(tot_summary_left[key])

    if config_file['channels_on_calib_summary_plots']:

        valid_channels = [i for i in range(6) if i not in config_file['skip_channels']]
        # f.e. if skipping 3,4: [0,1,2,5]

        for i_plot in range(len(config_file['channels_on_calib_summary_plots'])):

            if any(i in config_file['channels_on_calib_summary_plots'][i_plot] for i in config_file['skip_channels']):
                print(f"Warning: Channel requested on calibration summary plot is also marked for skipping. Check your config file! Skipping this plot.")
                continue

            # config_file['channels_on_calib_summary_plots'][i_plot] has actual indices for channels. However, tot_summary is indexed from 0 sequentially. We need to reindex based on the missing channels
            channels_for_plot_reindexed_by_missing = []
            for acutal_channel_index in config_file['channels_on_calib_summary_plots'][i_plot]:
                channels_for_plot_reindexed_by_missing.append( valid_channels.index(acutal_channel_index) )


            # threshold equality check  (if not, put each in legend, if all equal, just write it once)
            th_eq = tot_summary['tot_summary_text'][   channels_for_plot_reindexed_by_missing[0]    ]['th']
            for i in channels_for_plot_reindexed_by_missing:
                if tot_summary['tot_summary_text'][i]['th'] != th_eq:
                    th_eq = False
                    break

            fig = plt.figure(2, figsize=(15, 10))
            ax = fig.add_subplot(111)
            colors=["blue","darkred","black","darkgreen","purple","darkorange"]
            colors_dots=['dodgerblue',"red","grey","green","magenta","orange"]
            ax.set_title("FBCM", loc="left")
            ax.set_title(f"board{config_file['board_number']}, RC{config_file['RC']}", loc="right")
            ax.set_xlabel('Input charge [fC]')
            ax.set_ylabel('Time [ns]')
            for i in channels_for_plot_reindexed_by_missing:
                label_i = "Channel "+str(tot_summary['tot_summary_text'][i]['ch'])
                if not th_eq:
                    label_i += ", threshold: "+str(tot_summary['tot_summary_text'][i]['th'])+" fC"
                ax.plot(tot_summary['tot_summary_x'][i], tot_summary['tot_summary'][i], 's', color=colors[valid_channels[i]])
                ax.errorbar(tot_summary['tot_summary_x'][i], tot_summary['tot_summary'][i], yerr=tot_summary['tot_std_summary'][i], capsize=3, fmt=".", ecolor = colors[valid_channels[i]], color=colors_dots[valid_channels[i]], label=label_i) 
            goodlim_x = ax.get_xlim()
            goodlim_y = ax.get_ylim()
            for i in channels_for_plot_reindexed_by_missing:
                if "log" in config_file['selected_calibration_type']:
                    ax.plot(tot_summary['fit_chcalval'][i], tot_func(tot_summary['fit_chcalval'][i], *tot_summary['tot_summary_params'][i]) , colors[valid_channels[i]])

                if "linear" in config_file['selected_calibration_type']:
                    x_range = np.linspace(goodlim_x[0], goodlim_x[1],1000)
                    ax.plot(x_range, tot_summary['tot_summary_linear'][i](x_range) , colors[valid_channels[i]])

                if "spline" in config_file['selected_calibration_type']:
                    x_range = np.linspace(goodlim_x[0], goodlim_x[1],1000)
                    ax.plot(x_range, tot_summary['tot_summary_spline'][i](x_range) , colors[valid_channels[i]])


            current_xtics_major = ax.get_xticks()
            current_ytics_major = ax.get_yticks()
            ax.set_xticks(np.arange(current_xtics_major[0],current_xtics_major[-1],1))
            ax.set_xticks(np.arange(current_xtics_major[0],current_xtics_major[-1],config_file['summary_plots_xticks_density']),minor=True)
            ax.set_yticks(np.arange(current_ytics_major[0],current_ytics_major[-1],1)) #,minor=True)

            ax.grid(linestyle = "-")
            ax.grid(linestyle = "--", which='minor')

            ax.set_xlim(goodlim_x)
            ax.set_ylim(goodlim_y)

            if th_eq:
                ax.legend(title="Threshold: "+str(th_eq)+" fC",loc='best')
            else:
                ax.legend(loc='best') #, prop={'size': 10})  # bbox_to_anchor=(0, 0, 0.9, 0.9)

            fnameindex = "ch"+str( config_file['channels_on_calib_summary_plots'][i_plot][0] )
            for j in range(1,len(config_file['channels_on_calib_summary_plots'][i_plot]),1):
                fnameindex += "-"+str(config_file['channels_on_calib_summary_plots'][i_plot][j])

            fig.savefig(config_file['output_dir']+'/calib_plots/tot_tw_sample_%s_rccal_%d_summary_%s.jpg' % (sample, config_file['RC'], fnameindex), format='jpg', bbox_inches='tight')
            plt.close()

    
    return output



if __name__ == "__main__":
    import yaml
    import argparse
    import sys
    def parse_args(argv: list[str]) -> argparse.Namespace:
        parser = argparse.ArgumentParser()
        parser.add_argument('configfile_name')
        parser.add_argument("-w", "--show_warnings", action="store_true")
        return parser.parse_args(argv)
    args = parse_args(sys.argv[1:])

    with open(args.configfile_name, 'r') as file:
        config_file = yaml.safe_load(file)


    if not os.path.exists(str(config_file['output_dir'])+"/calib_plots"):
        os.makedirs(str(config_file['output_dir'])+"/calib_plots")

    set_warnings(bool(args.show_warnings))


    output = get_board_calibration(config_file)
    #print(output)
    #import code
    #code.interact(local=locals())