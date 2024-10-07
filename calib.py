import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
import math

warning_suppress = False
def hide_warnings(b):
    global warning_suppress
    warning_suppress = b
    if warning_suppress:
        print("Info: RuntimeWarnings have been suppressed!")

def get_voltage(dac_map, code):
    for i in dac_map:
        if code == i['code']:
            return i['voltage']
    raise ValueError("Missing code: %s in the dac map." % code)

def tot_func(x, arg0,arg1,arg2,arg3): # *args):
    '''
    eq_border = 4
    eq1 = x < eq_border
    eq2_b = args[0] + args[1]*np.sqrt(eq_border)  + args[2]*eq_border

    res = eq2_b + args[3]*(x-eq_border)
    res[eq1] = args[0] + args[1]*np.sqrt(x[eq1])  + args[2]*x[eq1] #-args[2]
    return res
    '''
    #return arg0 + arg1*np.log(x-arg2) + arg3* (np.log(x-arg4))**2
    
    if warning_suppress:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return arg0 + arg1*np.log(arg2*x+arg3)
    else:
        return arg0 + arg1*np.log(arg2*x+arg3)

    

def invert_tot_func(x, a,b,c,d):
    return -np.exp(-a/b)*(d*np.exp(a/b) - np.exp(x/b)) / c

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


    for ch_num in test_conditions['asic_channels']:

        toa =     results['toa'][ch_num].copy()
        tot =     results['tot'][ch_num].copy()

        bUseSigma = False
        try:
            tot_std = results['tot_std'][ch_num].copy()
            bUseSigma = True
        except:
            print("Warning: no tot_std key found in results! You are running on old data! (And fits may fail)")

        tw = twalk[ch_num].copy()
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

                # sinve v13 tests:
                #if (test_conditions['th_list_fC'][th_level_fC] not in test_conditions['th_list_per_rccal'][test_conditions['rccal'][r]]):
                #    continue

                fig = plt.figure()
                ax = fig.add_axes([0,0,1,1])
                ax.set_title('Timing scan (S: %s, CH: %s, RC=%x, th=%.2f fC)' % (sample, ch_num, test_conditions['rccal'][r], test_conditions['th_list_fC'][th_level_fC]))
                ax.set_ylabel('Time [ns]')
                ax.set_xlabel('Input charge [fC]')
                # start = np.argmin(err[r,0:256:step])*step # find first False what means first valid value
                if 'start' in test_conditions:
                    start = test_conditions['start']
                    if not isinstance(start, int) and not isinstance(start, np.int64):  
                        start = int(test_conditions['start'][ch_num,r,th_level_fC])
                elif 'start' in results:
                    start = results['start']
                    if not isinstance(start, int) and not isinstance(start, np.int64):
                        start = int(results['start'][ch_num,r,th_level_fC])
                else:
                    start = 0
                end = 255
                # print(chcalval[start:end:step]*1e15)
                #print(start,end,step)
                tw[r,th_level_fC,start:end:step] =  tw[r,th_level_fC,start:end:step] - np.min(tw[r,th_level_fC,start:end:step])

                myTot = tot[r,th_level_fC,start:end:step]

                th_level_fC_val = test_conditions['th_list_fC'][th_level_fC]
                #print(th_level_fC_val)
                mychcalval = np.concatenate(([th_level_fC_val*1e-15],chcalval[start:end:step]))


                if bUseSigma:
                    sigma_fit = tot_std[r,th_level_fC,start:end:step]*1e9
                else:
                    sigma_fit = np.array( [1] * len(myTot) )

                #pfit = curve_fit(tot_func, mychcalval*1e15, mytot_fit*1e9, p0=(11,2,0.01,1), sigma = mytot_sigma*1e9, absolute_sigma = True)
                popt, pcov = curve_fit(tot_func, chcalval[start:end:step]*1e15, myTot*1e9, p0=config_file['log_p0'], sigma = sigma_fit, absolute_sigma = True)
                # , p0=(11,2,0.01,1)

                # pfit = np.polyfit(mychcalval*1e15, myTot*1e9, 3)
                # print(pfit)

                #print("Convertin 8 to: "+str(tot_func(8.0, *popt)))

                perr = np.sqrt(np.diag(pcov))

                output['fitparams'].append(popt.copy())
                output['RC'].append(test_conditions['rccal'][r])
                output['th'].append(test_conditions['th_list_fC'][th_level_fC])
                output['channel'].append(ch_num)


                fit_chcalval = np.arange(0.6e-15, mychcalval[-1], (mychcalval[1]-mychcalval[0])/10)*1e15
                fit_tot = tot_func(fit_chcalval, *popt)
                # print(myTot[0])
                # fit_tot = tot_func(fit_chcalval, *(11,2,0.01))
                # fit_tot = tot_func(fit_chcalval, *(40e-9,-1e-9,1e-9))

                # fit_tot = pfit[0] + pfit[1]*fit_chcalval + pfit[2]*fit_chcalval**2 + pfit[3]*fit_chcalval**3
                # fit_tot = tot_func(fit_chcalval, *(myTot[0],2e-15,0.1))
                # print(pfit[0])
                # print(fit_tot[])

                from scipy.interpolate import CubicSpline
                splinex = chcalval[start:end:step]*1e15
                spliney = tot[r,th_level_fC,start:end:step]*1e9
                splinefit = CubicSpline(splinex, spliney)
                x_range = np.linspace(splinex[0], splinex[-1],1000)

                from scipy.interpolate import splrep, BSpline
                smoothing = config_file['spline_smoothing_param']
                tck_s = splrep(splinex, spliney, s=smoothing)

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
                ax.plot(chcalval[start:end:step]*1e15, tot[r,th_level_fC,start:end:step]*1e9, 'sb', label='ToT')
                if bUseSigma:
                    ax.errorbar(chcalval[start:end:step]*1e15, tot[r,th_level_fC,start:end:step]*1e9, yerr=tot_std[r,th_level_fC,start:end:step]*1e9, capsize=3, fmt=".", ecolor = "black")
                ax.plot(chcalval[start:end:step]*1e15, tw[r,th_level_fC,start:end:step]*1e9, 'or', label='TW')

                goodlim_x = ax.get_xlim()
                goodlim_y = ax.get_ylim()

                #params = []
                #for i in range(len(pfit[0])):
                #    params.append("%.2f ± %.2f" % (pfit[0][i], perr[i])   )

                params = ["%.2f" % p for p in popt]
                ax.plot(fit_chcalval, fit_tot, 'g', label='ToT fit %s' %  params)
                #ax.plot(x_range, splinefit(x_range), 'orange', label='ToT spline')
                ax.plot(x_range, BSpline(*tck_s)(x_range), 'purple', label=f'ToT spline (s={smoothing})')

                ax.set_xlim(goodlim_x)
                ax.set_ylim(goodlim_y)

                #fig.legend(loc=[0.14,0.82], prop={'size': 10})
                ax.legend(loc='best', prop={'size': 10})  # bbox_to_anchor=(0, 0, 0.9, 0.9)
                # plt.yscale('log')
                fig.savefig(config_file['output_dir']+'/calib_plots/tot_tw_sample_%s_channel_%d_rccal_%d_th_%s.jpg' % (sample, ch_num, test_conditions['rccal'][r],  test_conditions['th_list_fC'][th_level_fC]), format='jpg', bbox_inches='tight')
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
        parser.add_argument("-s", "--suppress_warnings", action="store_true")
        return parser.parse_args(argv)
    args = parse_args(sys.argv[1:])

    with open(args.configfile_name, 'r') as file:
        config_file = yaml.safe_load(file)


    if args.suppress_warnings:
        hide_warnings(True)


    output = get_board_calibration(config_file)
    #import code
    #code.interact(local=locals())