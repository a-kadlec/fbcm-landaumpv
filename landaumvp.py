import argparse
from collections import defaultdict
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy.optimize import curve_fit 
import struct
import sys
import os
from landaupy import landau, langauss
import calib
import yaml
import math
import glob
import warnings
from scipy.interpolate import BSpline

def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('configfile_name')
    parser.add_argument("-w", "--show_warnings", action="store_true")
    parser.add_argument("-t", "--timestamps", nargs="+", type=lambda x: datetime.strptime(x, "%a %b %d %H:%M:%S %p %Z %Y"))
    return parser.parse_args(argv)


def main(argv: list[str] = sys.argv[1:]):
    args = parse_args(argv)


    with open(args.configfile_name, 'r') as file:
        config_file = yaml.safe_load(file)

    if not os.path.exists(str(config_file['output_dir'])+"/calib_plots"):
        os.makedirs(str(config_file['output_dir'])+"/calib_plots")

    calib.set_warnings(bool(args.show_warnings))

    calibration = calib.get_board_calibration(config_file)

    x_axes_data, y_axes_data = _process_input_files(config_file, args.timestamps)

    output_dir = Path(config_file['output_dir'])
    print("Raw ToT plot...")
    raw_tot_plot(x_axes_data, y_axes_data, output_dir/f"tot_raw_board{config_file['board_number']}_{config_file['measurement']}_rc{config_file['RC']}.png", config_file)
    #print("Charge plot [log calib]...")
    #landaumvp_fit(x_axes_data, y_axes_data, output_dir/f"tot_fc_fit_log-cal.png", config_file, calibration, "log")
    #print("Charge plot [spline calib]...")
    #landaumvp_fit(x_axes_data, y_axes_data, output_dir/f"tot_fc_fit_spline-cal.png", config_file, calibration, "spline")
    #print("Common Charge plot [log vs pline calib]...")
    #landaumvp_fit(x_axes_data, y_axes_data, output_dir/f"tot_fc_fit.png", config_file, calibration, "both")
    print("Refined Landau plot...")
    landaumvp_refined_fit(x_axes_data, y_axes_data, output_dir/f"tot_fc_fit-refined_board{config_file['board_number']}_{config_file['measurement']}_rc{config_file['RC']}.png", config_file, calibration, config_file['used_calibration_type'])


def get_histograms(data: bytes, timestamps: list = []) -> dict[str, list]:
    histograms = defaultdict(lambda: ([], []))

    idx = 0
    while idx < len(data):
        timestamp, name_len, data_len = struct.unpack("dII", data[idx:idx+struct.calcsize("dII")])
        idx += struct.calcsize("dII")

        fmt = f"{name_len}s{data_len}I"
        hist_name, *hist_data = struct.unpack(fmt, data[idx:idx+struct.calcsize(fmt)])
        idx += struct.calcsize(fmt)
        try:
            if timestamp < timestamps[0]:
                continue
            if timestamp > timestamps[1]:
                continue
        except TypeError:
            pass
        if len(hist_data) == 0:
            print(f"WARNING: Read {hist_name} histogram, timestamp {timestamp}, with len {len(hist_data)}", file=sys.stderr)
            continue
        histograms[hist_name.decode()][0].append(timestamp)
        histograms[hist_name.decode()][1].append(np.asarray(hist_data[9:-2])) # strip header and trailer

    for hist_name in histograms.keys():
        histograms[hist_name] = (np.asarray(histograms[hist_name][0]), np.asarray(histograms[hist_name][1]))

    return histograms


def _clean_histograms_arrays(timestamps, hist_data):
    if not len(timestamps) % 2:
        timestamps = timestamps[1:]
        hist_data = hist_data[1:]

    # ensure hists[0] is interspill
        if (timestamps[1] - timestamps[0]) < (timestamps[2] - timestamps[1]):
            timestamps = timestamps[1:-1]
            hist_data = hist_data[1:-1]

    return timestamps, hist_data


def _get_intra_and_interspill_hists(timestamps, hist_data):
    #print(f"timestamps = {timestamps-timestamps[0]}")
    time_deltas = timestamps[1:] - timestamps[:-1]
    scale_factors = time_deltas[1::2]/time_deltas[::2]
    scaled_noise = scale_factors[:, None] * hist_data[1::2]
    #print(f"time_deltas = {time_deltas}")

    return sum(hist_data[2::2]), sum(scaled_noise)


def _mpv_to_thickness(mpv):
    return mpv * 1e-15 / 1.6e-19 / 78

def gauss(x, H, A, x0, sigma): 
    return H + A * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))


def _process_input_files(config_file, argtimestamps):

    histograms_data_list = []

    if isinstance(config_file['histogram_data_files'], str) or isinstance(config_file['histogram_data_files'], int):
        config_file['histogram_data_files'] = [config_file['histogram_data_files']]


    print("Processing "+str(len(config_file['histogram_data_files']))+" histogram data files:")
    for filename in config_file['histogram_data_files']:
        if isinstance(filename, str) and ".dat" in filename: # filename is full name, open that
            file_to_open = config_file['histogram_data_dir']+"/"+filename
        else: # filename is just run number, get file based on just that
            file_to_open = glob.glob(glob.escape(config_file['histogram_data_dir']) + "/*_run_"+str(filename)+"_*")
            if not file_to_open:
                raise FileNotFoundError("Check the syntax of your histogram_data_files, cannot find file with specified run number: "+str(filename))
            if len(file_to_open) > 1:
                raise Exception("Ambiguity error in histogram_data_files: more than one files found for run number "+str(filename)+", specify exact file name instead")
            
            file_to_open = file_to_open[0]

        print(file_to_open)
        with open(file_to_open, "rb") as in_f:
            histograms_data_list.append( get_histograms(in_f.read(), argtimestamps))



    y_axes = []
    x_axes = []
    for (hist_name, (timestamps, hist_data)), in zip(histograms_data_list[0].items()):
        timestamps, hist_data = _clean_histograms_arrays(timestamps, hist_data)

        channel_number = int(hist_name[-1])

        intraspill_sum, interspill_sum = _get_intra_and_interspill_hists(timestamps, hist_data)
        time_bins = np.arange(len(interspill_sum))/1.28
        bin_width = time_bins[-1]/len(time_bins)
        time_bins += bin_width/2

        x_axes.append(time_bins.copy())
        y_axes.append( abs(interspill_sum - intraspill_sum).copy())   # was intra - inter, reversed

    for i in range(1,len(histograms_data_list),1):
        for (hist_name, (timestamps, hist_data)), in zip(histograms_data_list[i].items()):
            timestamps, hist_data = _clean_histograms_arrays(timestamps, hist_data)

            channel_number = int(hist_name[-1])

            intraspill_sum, interspill_sum = _get_intra_and_interspill_hists(timestamps, hist_data)
            time_bins = np.arange(len(interspill_sum))/1.28
            bin_width = time_bins[-1]/len(time_bins)
            time_bins += bin_width/2

            if not np.array_equal(x_axes[channel_number], time_bins):
                warnings.warn("The time bins in histogram file "+config_file['histogram_data_files']+" do not match the first file, IGNORING FILE")
                break

            y_axes[channel_number] = y_axes[channel_number] + abs(interspill_sum - intraspill_sum) 
    
    return x_axes, y_axes




def raw_tot_plot(x_axes_data, y_axes_data, output_file, config_file):

    cutoff = config_file['cutoff_ns']
    axs = []
    bottom_axs = []
    fig = plt.figure(tight_layout=True, figsize=(15, 10))
    gs_top_level = plt.GridSpec(2, 3, figure=fig)
    for i in range(6):
        gs = gs_top_level[i].subgridspec(2, 1, height_ratios=[4, 1], hspace=0)
        axs.append(fig.add_subplot(gs[0]))
        bottom_axs.append(0)

    for channel_number in range(len(y_axes_data)):
        time_bins = x_axes_data[channel_number]
        y_axis = y_axes_data[channel_number]
        ax = axs[channel_number]
        bin_width = time_bins[-1]/len(time_bins)

        if cutoff:
            for i in range(len(time_bins)):
                if time_bins[i] > cutoff:
                    time_bins = time_bins[:i]
                    y_axis = y_axis[:i]
                    break
    
        errorbars = np.sqrt(y_axis)
        #print(y_axis)
        #print(errorbars)
        for i in range(len(errorbars)):
            if errorbars[i] == 0:
                errorbars[i] = 1

        #ax.plot(time_bins, y_axis, ".", color="black")   
        ax.plot(time_bins, y_axis, 'sb')
        ax.errorbar(time_bins, y_axis, yerr=errorbars, capsize=3, fmt=".", ecolor = "black")

        fig.suptitle(f"ToT distribution of board{config_file['board_number']} RC"+str(config_file['RC']))
        ax.set_title(f"Channel {channel_number}, th: {config_file['threshold_per_channel'][channel_number]} fC")
        ax.set_xticks(np.arange(math.floor(min(time_bins)), math.ceil(max(time_bins)), 4.0))
        ax.set_xticks(np.arange(math.floor(min(time_bins)), math.ceil(max(time_bins)), 2.0), minor=True)
        ax.set_ylabel(f"Count / {bin_width:.2f} ns")
        ax.set_xlabel("Time over threshold [ns]")
        ax.grid(linestyle = "--")

    fig.savefig(output_file)
    plt.close(fig)









def landaumvp_refined_fit(x_axes_data, y_axes_data, output_file, config_file, calibration, calib_type = "log"):

    cutoff = config_file['cutoff_ns']
    axs = []
    bottom_axs = []
    fig = plt.figure(tight_layout=True, figsize=(15, 10))
    gs_top_level = plt.GridSpec(2, 3, figure=fig)
    for i in range(6):
        gs = gs_top_level[i].subgridspec(2, 1, height_ratios=[4, 1], hspace=0)
        axs.append(fig.add_subplot(gs[0]))
        bottom_axs.append(0)


    mpvs = []
    mpv_errs = []
    mpv_systs_up = []
    mpv_systs_down = []
    with open(config_file['output_dir']+"/results_board"+str(config_file['board_number'])+"_"+config_file['measurement']+".dat", "w") as result_file:
        print("#Channel number | MPV [fC] | MPV stat [fC] | MPV syst up [fC] | MPV syst down [fC] | d_eff [um] | d_eff stat+syst up [um] | d_eff stat+syst down [um]", file=result_file)


        for channel_number in range(len(y_axes_data)):
            time_bins = x_axes_data[channel_number]
            y_axis = y_axes_data[channel_number]
            ax = axs[channel_number]
            bin_width = time_bins[-1]/len(time_bins)

            if cutoff:
                for i in range(len(time_bins)):
                    if time_bins[i] > cutoff:
                        time_bins = time_bins[:i]
                        y_axis[:i]
                        break
        

            # Conversion of x axis to fc
            # find calib for this setup
            for calib_i in range(len(calibration['RC'])): 
                if calibration['th'][calib_i] == config_file['threshold_per_channel'][channel_number]  and calibration['RC'][calib_i] == config_file['RC'] and calibration['channel'][calib_i] == channel_number:
                    break

            if calib_type == "log":
                fc_bins = calib.invert_tot_func(time_bins, *calibration['fitparams'][calib_i])
            if calib_type == "spline":
                fc_bins = BSpline(*calibration['spline'][calib_i])(time_bins)

            for i in range(len(fc_bins)):
                if fc_bins[i] > 0:  # to remove nans
                    fc_bins = fc_bins[i:]
                    y_axis[i:]
                    break

            for i in range(len(fc_bins)):
                if fc_bins[i] > config_file['cutoff_fC']:
                    fc_bins = fc_bins[:i]
                    y_axis = y_axis[:i]
                    break
            

            #ax.plot(fc_bins, y_axis, 'sb') 
            #ax.errorbar(fc_bins, y_axis, yerr=np.sqrt(y_axis), capsize=3, fmt=".", ecolor = "black")

            popt, pcov = curve_fit(lambda x, N, mpv, xi: N*landau.pdf(x, mpv, xi), fc_bins, y_axis, p0 = config_file['landau_p0'])

            # refine fit:
            fc_bins_fit = []
            y_axis_fit = []
            fc_bins_off = []
            y_axis_off = []
            fc_bins_left_off = []
            y_axis_left_off = []
            for i in range(len(y_axis)):
                if (fc_bins[i] < popt[1] or abs(y_axis[i] -  popt[0]*landau.pdf(fc_bins[i], popt[1], popt[2])) / y_axis[i] < (config_file['cutoff_ref']/100.)) and i >= config_file['cutoff_lower'][channel_number]:    # and y_axis[i] > 0 
                    fc_bins_fit.append(fc_bins[i])
                    y_axis_fit.append(y_axis[i])
                else:
                    if i >= config_file['cutoff_lower'][channel_number]:
                        fc_bins_off.append(fc_bins[i])
                        y_axis_off.append(y_axis[i])
                    else:
                        fc_bins_left_off.append(fc_bins[i])   # if studying points ignored from the left, they are kept separate for the syst study part
                        y_axis_left_off.append(y_axis[i])


            popt_ref, pcov_ref = curve_fit(lambda x, N, mpv, xi: N*landau.pdf(x, mpv, xi), fc_bins_fit, y_axis_fit, p0 = popt) #, sigma = np.sqrt(y_axis_fit), absolute_sigma = True)
            perr_ref = np.sqrt(np.diag(pcov_ref))

            # systematic study: how much does mpv change by adding an extra point from the cutoff ones
            # plus one point (if any were ignored)
            if len(fc_bins_off) > 0:
                fc_bins_syst = list(fc_bins_fit)
                fc_bins_syst.append(fc_bins_off[0])
                y_axis_syst = list(y_axis_fit)
                y_axis_syst.append(y_axis_off[0])
                popt_syst, pcov_syst = curve_fit(lambda x, N, mpv, xi: N*landau.pdf(x, mpv, xi), fc_bins_syst, y_axis_syst, p0 = popt)
                mpv_syst_more = popt_syst[1] - popt_ref[1]
            else:
                mpv_syst_more = 0

            # one point less
            fc_bins_syst = fc_bins_fit[:-1]
            y_axis_syst = y_axis_fit[:-1]
            popt_syst, pcov_syst = curve_fit(lambda x, N, mpv, xi: N*landau.pdf(x, mpv, xi), fc_bins_syst, y_axis_syst, p0 = popt)
            mpv_syst_less = popt_syst[1] - popt_ref[1]


            larger = mpv_syst_more if mpv_syst_more > mpv_syst_less else mpv_syst_less
            smaller = mpv_syst_more if mpv_syst_more <= mpv_syst_less else mpv_syst_less
            
            if larger > 0:
                mpv_systs_up.append(larger)
                if smaller > 0:
                    mpv_systs_down.append(0)
                else:
                    mpv_systs_down.append(abs(smaller))
            else:
                mpv_systs_up.append(0)
                mpv_systs_down.append(abs(smaller))


            # for plotting, join the off points arrays
            fc_bins_left_off.extend(fc_bins_off)
            fc_bins_off = fc_bins_left_off
            y_axis_left_off.extend(y_axis_off)
            y_axis_off = y_axis_left_off


            ax.plot(fc_bins_fit, y_axis_fit, 'sb') 
            ax.errorbar(fc_bins_fit, y_axis_fit, yerr=np.sqrt(y_axis_fit), capsize=3, fmt=".", ecolor = "black")
            if len(fc_bins_off) > 0:
                ax.plot(fc_bins_off, y_axis_off, '.', color="black") 
                ax.errorbar(fc_bins_off, y_axis_off, yerr=np.sqrt(y_axis_off), capsize=3, fmt="x", ecolor = "black", color="black")

            if len(fc_bins_off) > 0:
                x_axis_fit = np.linspace(min(fc_bins_fit) if min(fc_bins_fit) > math.floor(min(fc_bins_off)) else math.floor(min(fc_bins_fit)), max(fc_bins_fit), 1000)
                x_axis_off = np.linspace(math.floor(min(fc_bins_off)) if math.floor(min(fc_bins_off)) < max(fc_bins_fit) else max(fc_bins_fit), max(fc_bins_off), 1000)
            else:
                x_axis_fit = np.linspace(math.floor(min(fc_bins_fit)), max(fc_bins_fit), 1000)

            mpvs.append(popt_ref[1])
            mpv_errs.append(perr_ref[1])

            mpv_string = eval( " '%."+str(config_file['mpv_display_precision'])+"f±%."+str(config_file['mpv_display_precision'])+"f' % (popt_ref[1],perr_ref[1])"  
            + "+"+ "r'$^{+%."+str(config_file['mpv_display_precision'])+r"f}_{-%."+str(config_file['mpv_display_precision'])+r"f}$'  % (mpv_systs_up[-1],mpv_systs_down[-1])"    )

            if len(fc_bins_off) > 0:
                ax.plot(x_axis_off, popt_ref[0]*landau.pdf(x_axis_off, popt_ref[1], popt_ref[2]), "--", color="crimson")  
            ax.plot(x_axis_fit, popt_ref[0]*landau.pdf(x_axis_fit, popt_ref[1], popt_ref[2]), "-", color="crimson", label = f"MPV = ({mpv_string}) fC\n"+r"$d_{eff}$"+r" = (%.2f$^{+%.2f}_{-%.2f}$)" % (_mpv_to_thickness(popt_ref[1]) , _mpv_to_thickness(perr_ref[1])+_mpv_to_thickness(mpv_systs_up[-1]) , _mpv_to_thickness(perr_ref[1])+_mpv_to_thickness(mpv_systs_down[-1])     ) + r" $\mu$m")  
                    # "MPV=(%.2f±%.2f) fC" % (popt_ref[1],perr_ref[1]))  


            leg = ax.legend(loc='best', prop={'size': 11}, handletextpad=0, handlelength=0)
            #for item in leg.legendHandles:
            #    item.set_visible(False)
            fig.suptitle(f"Landau Distribution fits to Amplitude Spectrum of board{config_file['board_number']}, RC{config_file['RC']}, {calib_type}-cal.")

            ax.set_title(f"Channel {channel_number}, th: {calibration['th'][calib_i]} fC")
            ax.set_xticks(np.arange(math.floor(min(fc_bins)), math.ceil(max(fc_bins)), 2.0))
            ax.set_xticks(np.arange(math.floor(min(fc_bins)), math.ceil(max(fc_bins)), 1.0), minor=True)
            ax.set_ylabel(f"Count")
            ax.set_xlabel("Charge [fC]")
            ax.grid(linestyle = "--")

            print("Channel "+str(channel_number)+" eff. thickness: (%.2f±%.2f(stat)) um" % (_mpv_to_thickness(popt_ref[1]) , _mpv_to_thickness(perr_ref[1])))

            print(str(channel_number), popt_ref[1], perr_ref[1], mpv_systs_up[-1], mpv_systs_down[-1], _mpv_to_thickness(popt_ref[1]), _mpv_to_thickness(perr_ref[1])+_mpv_to_thickness(mpv_systs_up[-1]), _mpv_to_thickness(perr_ref[1])+_mpv_to_thickness(mpv_systs_down[-1]), file=result_file)


    fig.savefig(output_file)
    plt.close(fig)

    # Summary plot
    fig = plt.figure(2, figsize=(7, 5))
    ax = fig.add_subplot(111)
    ax.set_title(f"Summary plot of board{config_file['board_number']}, RC{config_file['RC']}, {calib_type}-cal.")
    ax.set_xlabel('Threshold [fC]')
    ax.set_ylabel('MPV [fC]')
    colors=["blue","crimson","darkorange","black","darkgreen","purple"]
    colors_dots=['dodgerblue',"red","orange","grey","green","magenta"]
    for i in range(len(config_file['threshold_per_channel'])):
        #asymmetric_error = np.array(list(zip(mpv_errs[i]+mpv_systs_down[i], mpv_errs[i]+mpv_systs_up[i]))).T
        asymmetric_error = [[mpv_errs[i]+mpv_systs_down[i]], [mpv_errs[i]+mpv_systs_up[i]]]

        ax.errorbar(config_file['threshold_per_channel'][i], mpvs[i], yerr=asymmetric_error, capsize=3, fmt=".", ecolor = "black", color="black")
        ax.plot(config_file['threshold_per_channel'][i], mpvs[i], 's', color=colors[i])
        ax.errorbar(config_file['threshold_per_channel'][i], mpvs[i], yerr=mpv_errs[i], capsize=3, fmt=".", ecolor = colors[i], color=colors_dots[i], label="Channel "+str(i))

    ax.grid(linestyle = "--")
    ax.legend(loc='best', prop={'size': 11})

    y1, y2 = ax.get_ylim()
    x1, x2 = ax.get_xlim()
    ax2 = ax.twinx()
    ax2.set_ylim(_mpv_to_thickness(y1),_mpv_to_thickness(y2))
    ax2.set_yticks( range(int(_mpv_to_thickness(y1)), int(_mpv_to_thickness(y2)), (int((_mpv_to_thickness(y2) - _mpv_to_thickness(y1))/15))))
    ax2.tick_params(axis="y", which = 'major', labelsize=10)
    ax2.set_ylabel(r"$d_{eff}$ [$\mu$m]", fontsize = 11)
    ax2.set_xlim(x1,x2)

    fig.savefig(config_file['output_dir']+f"/MPV_th_summary_board{config_file['board_number']}_{config_file['measurement']}_rc{config_file['RC']}.png")





# OLD FUNCTION: this was used for checking if landau or landau-gauss convolution is the better fit.
def landaumvp_fit(x_axes_data, y_axes_data, output_file, config_file, calibration, calib_type = "log"):

    cutoff = config_file['cutoff_ns']
    axs = []
    bottom_axs = []
    fig = plt.figure(tight_layout=True, figsize=(18, 12))
    gs_top_level = plt.GridSpec(2, 3, figure=fig)
    for i in range(6):
        gs = gs_top_level[i].subgridspec(2, 1, height_ratios=[4, 1], hspace=0)
        axs.append(fig.add_subplot(gs[0]))
        bottom_axs.append(0)


    for channel_number in range(len(y_axes_data)):
        time_bins = x_axes_data[channel_number]
        y_axis = y_axes_data[channel_number]
        ax = axs[channel_number]
        bin_width = time_bins[-1]/len(time_bins)

        if cutoff:
            for i in range(len(time_bins)):
                if time_bins[i] > cutoff:
                    time_bins = time_bins[:i]
                    y_axis[:i]
                    break
    

        # Conversion of x axis to fc
        # find calib for this setup
        for calib_i in range(len(calibration['RC'])): 
            if calibration['th'][calib_i] == config_file['threshold_per_channel'][channel_number]  and calibration['RC'][calib_i] == config_file['RC'] and calibration['channel'][calib_i] == channel_number:
                break

        if calib_type == "log":
            fc_bins = calib.invert_tot_func(time_bins, *calibration['fitparams'][calib_i])
        if calib_type == "spline":
            fc_bins = BSpline(*calibration['spline'][calib_i])(time_bins)
        else:
            # do both
            fc_bins = calib.invert_tot_func(time_bins, *calibration['fitparams'][calib_i])
            fc_bins2 = BSpline(*calibration['spline'][calib_i])(time_bins)
            y_axis2 = y_axis.copy()
            for i in range(len(fc_bins2)):
                if fc_bins2[i] > 0:
                    fc_bins2 = fc_bins2[i:]
                    y_axis2[i:]
                    break
            for i in range(len(fc_bins2)):
                if fc_bins2[i] > config_file['cutoff_fC']:
                    fc_bins2 = fc_bins2[:i]
                    y_axis2 = y_axis2[:i]
                    break    

        for i in range(len(fc_bins)):
            if fc_bins[i] > 0:
                fc_bins = fc_bins[i:]
                y_axis[i:]
                break

        for i in range(len(fc_bins)):
            if fc_bins[i] > config_file['cutoff_fC']:
                fc_bins = fc_bins[:i]
                y_axis = y_axis[:i]
                break
        

        #ax.plot(fc_bins, y_axis, ".", color="black")    
        ax.errorbar(fc_bins, y_axis, yerr=np.sqrt(y_axis), capsize=3, fmt=".", ecolor = "black")

        popt, pcov = curve_fit(lambda x, N, mpv, xi: N*landau.pdf(x, mpv, xi), fc_bins, y_axis, p0 = config_file['landau_p0'])
        perr = np.sqrt(np.diag(pcov))

        popt2, pcov2 = curve_fit(lambda x, N, mpv, xi, sigma: N*langauss.pdf(x, mpv, xi, sigma), fc_bins, y_axis, p0 = [popt[0],popt[1],popt[2],1])
        perr2 = np.sqrt(np.diag(pcov2))

        if calib_type == "both":
            ax.plot(fc_bins2, y_axis2, ".", mfc='none', mec='blue')    

            popt_landau_2, pcov_landau_2 = curve_fit(lambda x, N, mpv, xi: N*landau.pdf(x, mpv, xi), fc_bins2, y_axis2, p0 = config_file['landau_p0'])
            perr_landau_2 = np.sqrt(np.diag(pcov_landau_2))



        x_axis = np.linspace(math.floor(min(fc_bins)), max(fc_bins), 1000)

        if calib_type != "both":
            ax.plot(x_axis, popt2[0]*langauss.pdf(x_axis, popt2[1], popt2[2], popt2[3]), "-", color="red", label = "LandauXGauss MPV=(%.2f±%.2f) fC" % (popt2[1],perr2[1]))    
            ax.plot(x_axis, popt[0]*landau.pdf(x_axis, popt[1], popt[2]), "--", color="blue", label = "Landau MPV=(%.2f±%.2f) fC" % (popt[1],perr[1]))    
        else:
            ax.plot(x_axis, popt[0]*landau.pdf(x_axis, popt[1], popt[2]), "-", color="red", label = "[log-cal] MPV=(%.2f±%.2f) fC" % (popt[1],perr[1]))   
            x_axis2 = np.linspace(math.floor(min(fc_bins2)), max(fc_bins2), 1000)
            ax.plot(x_axis2, popt_landau_2[0]*landau.pdf(x_axis2, popt_landau_2[1], popt_landau_2[2]), linestyle=(0, (5, 10)), color="blue", label = "[spline-cal] MPV=(%.2f±%.2f) fC" % (popt_landau_2[1],perr_landau_2[1]))   


        ax.legend(loc='best', prop={'size': 11})
        if calib_type == "both":
            fig.suptitle(f"Distribution fits to Amplitude Spectrum of board{config_file['board_number']}, RC{config_file['RC']}, Landau fits")
        else:
            fig.suptitle(f"Distribution fits to Amplitude Spectrum of board{config_file['board_number']}, RC{config_file['RC']}, {calib_type}-cal.")

        ax.set_title(f"Channel {channel_number}, th: {calibration['th'][calib_i]} fC")
        ax.set_xticks(np.arange(math.floor(min(fc_bins)), math.ceil(max(fc_bins)), 2.0))
        ax.set_xticks(np.arange(math.floor(min(fc_bins)), math.ceil(max(fc_bins)), 1.0), minor=True)
        ax.set_ylabel(f"Count")
        ax.set_xlabel("Charge [fC]")
        ax.grid(linestyle = "--")
    fig.savefig(output_file)
    plt.close(fig)







if __name__ == "__main__":
    main()
