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
    if isinstance(config_file['calibration_types'], str):
        bCalibLog = "log" in config_file['calibration_types']
        bCalibSpline = "spline" in config_file['calibration_types']
    elif isinstance(config_file['calibration_types'], list):
        for string in config_file['calibration_types']:
            if "log" in string:
                bCalibLog = True
            if "spline" in string:
                bCalibSpline = True

    # sanity check
    if not bCalibSpline and not bCalibLog:
        raise Exception("Error: No calibration type specified, check your config file.")
    if config_file['used_calibration_type'] == "log" and not bCalibLog:
        raise Exception("Error: You're trying to use log calibration for the histogram data, but did not specify log as an option for the calibration data. Check your config file.")
    if config_file['used_calibration_type'] == "spline" and not bCalibSpline:
        raise Exception("Error: You're trying to use spline calibration for the histogram data, but did not specify spline as an option for the calibration data. Check your config file.")



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
    output['RC'] = []
    output['th'] = []
    output['channel'] = []

    try:
        testboard_channels = test_conditions["testboard_channels"]
        asic_channels = test_conditions["asic_channels"]
    except KeyError:
        testboard_channels = test_conditions["channels"]
        asic_channels = test_conditions["channels"]

    #for testboard_channel in test_conditions['channels']:
    for asic_channel, testboard_channel in zip(asic_channels,testboard_channels):

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

                fig = plt.figure()
                ax = fig.add_axes([0,0,1,1])
                ax.set_title('Timing scan (S: %s, CH: %s, RC=%x, th=%.2f fC)' % (sample, testboard_channel, test_conditions['rccal'][r], test_conditions['th_list_fC'][th_level_fC]))
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
                else:
                    sigma = np.array( [1] * len(myTot) )

                
                # issue: if x value is < 1, log fit will fail - exclude these points from the fit.
                x_off = []
                tot_off = []
                sigma_off = []
                for find_i in range(len(xvalues)):
                    if xvalues[find_i] >= 0.0:
                        xfit = xvalues[find_i:]
                        myTot_fit = myTot[find_i:]
                        sigma_fit = sigma[find_i:]

                        if find_i > 0:
                            x_off = xvalues[:find_i]
                            tot_off = myTot[:find_i]
                            sigma_off = sigma[:find_i]
                        break
                
                if bCalibLog:
                    #xfit_trick = np.array(xfit) + 5
                    popt, pcov = curve_fit(tot_func, xfit, myTot_fit, p0=config_file['log_p0'], sigma = sigma_fit, absolute_sigma = True)
                    #popt[2] += 5

                    Nexp = tot_func(xfit, *popt)
                    Ndif = myTot_fit - Nexp
                    chi2 = np.sum((Ndif/sigma_fit)**2)
                    chi2ondf = chi2 / (len(myTot_fit) - len(popt))

                    output['fitparams'].append(popt.copy())


                    fit_chcalval = np.arange(0.6e-15, mychcalval[-1], (mychcalval[1]-mychcalval[0])/10)*1e15
                    fit_tot = tot_func(fit_chcalval, *popt)

                if bCalibSpline:
                    from scipy.interpolate import CubicSpline
                    splinex = chcalval[start:end:step]*1e15
                    spliney = tot[r,th_level_fC,start:end:step]*1e9
                    splinefit = CubicSpline(splinex, spliney)
                    x_range = np.linspace(splinex[0], splinex[-1],1000)

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

                    output['spline'].append(tck_s_inv)
                #print(chcalval[start]*1e15)
                #ax.plot(chcalval[start:end:step]*1e15, toa[r,start:end:step]*1e9, 'sr', label='ToA')

                if x_off:
                    ax.errorbar(x_off, tot_off, yerr=sigma_off, capsize=3, fmt="x", ecolor = "black", color="black")
                ax.plot(xfit, myTot_fit, 'sb', label='ToT')
                if bUseSigma:
                    ax.errorbar(xfit, myTot_fit, yerr=sigma_fit, capsize=3, fmt=".", ecolor = "black")
                ax.plot(xvalues, tw[r,th_level_fC,start:end:step]*1e9, 'or', label='TW')

                goodlim_x = ax.get_xlim()
                goodlim_y = ax.get_ylim()

                #params = []
                #for i in range(len(pfit[0])):
                #    params.append("%.2f ± %.2f" % (pfit[0][i], perr[i])   )

                if bCalibLog:
                    params = ["%.2f" % p for p in popt]
                    ax.plot(fit_chcalval, fit_tot, 'g', label='ToT fit %s' %  params +r", $\chi^{2}$/dof="+"{:.1f}".format(chi2ondf))
                #ax.plot(x_range, splinefit(x_range), 'orange', label='ToT spline')
                if bCalibSpline:
                    ax.plot(x_range, BSpline(*tck_s)(x_range), 'purple', label=f'ToT spline (s={smoothing})'+r", $\chi^{2}$/dof="+"{:.1f}".format(chi2ondf_sp))

                ax.set_xlim(goodlim_x)
                ax.set_ylim(goodlim_y)

                #fig.legend(loc=[0.14,0.82], prop={'size': 10})
                ax.legend(loc='best', prop={'size': 10})  # bbox_to_anchor=(0, 0, 0.9, 0.9)
                # plt.yscale('log')
                fig.savefig(config_file['output_dir']+'/calib_plots/tot_tw_sample_%s_channel_%d_rccal_%d_th_%s.jpg' % (sample, testboard_channel, test_conditions['rccal'][r],  test_conditions['th_list_fC'][th_level_fC]), format='jpg', bbox_inches='tight')
                plt.close()
    return output




def get_board_calibration(config_file):
    matplotlib.rc('font', size=16)
    
    
    samplenum = config_file['board_number']
    meas = "_"+config_file['measurement']
    sample = str(samplenum) + meas

    if config_file['calibration_data_left']:
        sample = sample + "_R"

    data = process_npz(dict(np.load(config_file['calibration_data'], allow_pickle=True, encoding="ASCII")))

    output = get_ASIC_calibration(data, sample, config_file)

    if config_file['calibration_data_left']:
        data_left = process_npz(dict(np.load(config_file['calibration_data_left'], allow_pickle=True, encoding="ASCII")))
        sample_left = str(samplenum) + meas + "_L"
        output_left = get_ASIC_calibration(data_left,sample_left, config_file)

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

    #print(output)
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


    set_warnings(bool(args.show_warnings))


    output = get_board_calibration(config_file)
    #print(output)
    #import code
    #code.interact(local=locals())