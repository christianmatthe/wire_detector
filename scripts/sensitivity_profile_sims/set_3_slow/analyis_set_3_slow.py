import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as mpatches
import os
import dill
import sys
import scipy as sp
from scipy import optimize
from scipy.interpolate import interp1d

from Wire_detector import Wire
from time import time
import pandas as pd

#plot Options
import matplotlib as mpl
font = {#'family' : 'normal','weight' : 'bold',
        'size'   : 16
        #,'serif':['Helvetica']
        }
mpl.rc('font', **font)
#################################
def plot_signal_over_x_off(signal_arr, x_arr, 
                            plotname = "signal_over_x_off", power = 100):
    fig = plt.figure(0, figsize=(8,6.5))
    ax1=plt.gca()
    ax1.plot(x_arr * 1e3, signal_arr, label = f"{power}µW")

    ax1.set_xlabel(r"x_offset_beam [mm]")
    ax1.set_ylabel(r"A.U.")

    plt.legend(shadow=True)
    plt.tight_layout()
    plt.grid(True)

    format_im = 'png' #'pdf' or png
    dpi = 300
    plt.savefig(plot_dir + plotname
                + '.{}'.format(format_im),
                format=format_im, dpi=dpi)
    ax1.cla()
    fig.clf()
    plt.close()



if __name__ == "__main__":
    top_dir = os.path.dirname(os.path.abspath(__file__)) + os.sep

    plot_dir = top_dir + "analysis_2023_06_13/".format()
    os.makedirs(plot_dir, exist_ok=True)
    heat_flow_dir = top_dir + "heat_flow/"

    ##### Constants
    E_recombination = Wire().E_recombination
    n_wire_elements  = 100
    segment_length = Wire(n_wire_elements = n_wire_elements,
                          l_wire=2*10**-2).l_segment
    emissivity = 0.08
    pressure = 0
    d = 5
    i_current = 1
    l_wire = 2
    ####

    ### scan lists
    x_pos_list = np.linspace(-1,0, 40, endpoint= True) + segment_length/2
    # print(x_pos_list)
    power_list = [100.0, 1.0] # in µW
    #power = 100.0
    ###
    ### fix position errors,  that stem from misalignement of x_pos_list and
    # wire segment positions
    # Have to correct for 1e2 factor due to units in  which l_segment is stored
    # in the Wire object
    segment_positions = [((i + 0.5) * segment_length*1e2 - (l_wire / 2)) 
                         for i in range(n_wire_elements)]
    # print("segment_positions", segment_positions)
    # print("x_pos_list", x_pos_list)
    # find  closest correspodng segment
    x_pos_seg = np.array([min(segment_positions, key=lambda x:abs(x-x_pos)) 
                  for x_pos in x_pos_list]) *1e-2
    print("x_pos_seg", list(x_pos_seg))



    for power in power_list:
        # Initialize Arrays
        x_arr = np.zeros(( len(x_pos_list)))
        U_arr = np.zeros(( len(x_pos_list)))
        T_max_arr = np.zeros(( len(x_pos_list)))
        T_avg_arr = np.zeros(( len(x_pos_list)))
        signal_arr = np.zeros(( len(x_pos_list)))
        R_arr = np.zeros(( len(x_pos_list)))
        for n_i, x_pos in enumerate(x_pos_list):
            run_name = "lw_{}_i_1".format(l_wire,i_current)
            #TODO Include enumerates, output plots for every i_current
            wire = Wire()
            
            wire = wire.load(top_dir + "{}muW_x_off_{}\\".format(
                            power, n_i) 
                            + "results\\" + run_name)
            wire.gen_k_heat_cond_function()
            #l_beam = wire.l_beam

            U_beam_off = wire.U_wire(0)
            U_beam_on = wire.U_wire(-1)
            
            U_delta = U_arr[n_i] = U_beam_on - U_beam_off
            signal = signal_arr[n_i] = U_delta / U_beam_off

            T_max = T_max_arr[n_i] = np.amax(
                wire.record_dict["T_distribution"][-1])
            T_avg = T_avg_arr[n_i] = np.average(
                wire.record_dict["T_distribution"][-1])

            R_arr[n_i] = wire.resistance_total()
            x_arr[n_i] = wire.x_offset_beam

        # print("T_avg_arr: ", T_avg_arr)
        # print("T_max_arr: ", T_max_arr)
        # print("signal_arr: ", signal_arr)
        # print("x_arr: ", x_arr)

        #### Plot
            
        plot_signal_over_x_off(signal_arr, x_arr, 
                        plotname = f"signal_over_x_off_{power}µW",
                        power  = power)
        
        signal_arr_mir = np.concatenate([signal_arr, signal_arr[::-1]],
                                        axis=None)
        x_arr_mir = np.concatenate([x_arr, -1 * x_arr[::-1]],axis=None)
        # print("x_arr_mir: ", x_arr_mir)
        plot_signal_over_x_off(signal_arr_mir, x_arr_mir, 
                        plotname = f"signal_over_x_off_mirrored_{power}µW",
                        power  = power)
        
        signal_arr_norm = signal_arr_mir / np.max(signal_arr_mir)
        x_arr_mir = np.concatenate([x_arr, -1 * x_arr[::-1]],axis=None)
        # print("x_arr_mir: ", x_arr_mir)
        plot_signal_over_x_off(signal_arr_norm, x_arr_mir, 
                        plotname = f"signal_over_x_off_norm_{power}µW",
                        power  = power)

        signal_arr_norm = signal_arr_mir / np.max(signal_arr_mir)
        x_arr_mir = np.concatenate([x_arr, -1 * x_arr[::-1]],axis=None)
        # print("x_arr_mir: ", x_arr_mir)
        plot_signal_over_x_off(signal_arr_norm, 
                        np.concatenate((x_pos_seg, -1 * x_pos_seg[::-1])), 
                        plotname = f"signal_over_x_off_seg_{power}µW",
                        power  = power)
        signal_arr_norm_1 = signal_arr_norm[::]
        print("signal_arr_norm_1", list(signal_arr_norm_1))


##### HACK !!!!#######
    for power in [power_list[0]]:
        # Initialize Arrays
        x_arr = np.zeros(( len(x_pos_list)))
        U_arr = np.zeros(( len(x_pos_list)))
        T_max_arr = np.zeros(( len(x_pos_list)))
        T_avg_arr = np.zeros(( len(x_pos_list)))
        signal_arr = np.zeros(( len(x_pos_list)))
        R_arr = np.zeros(( len(x_pos_list)))
        for n_i, x_pos in enumerate(x_pos_list):
            run_name = "lw_{}_i_1".format(l_wire,i_current)
            #TODO Include enumerates, output plots for every i_current
            wire = Wire()
            
            wire = wire.load(top_dir + "{}muW_x_off_{}\\".format(
                            power, n_i) 
                            + "results\\" + run_name)
            wire.gen_k_heat_cond_function()
            #l_beam = wire.l_beam

            U_beam_off = wire.U_wire(0)
            U_beam_on = wire.U_wire(-1)
            
            U_delta = U_arr[n_i] = U_beam_on - U_beam_off
            signal = signal_arr[n_i] = U_delta / U_beam_off

            T_max = T_max_arr[n_i] = np.amax(
                wire.record_dict["T_distribution"][-1])
            T_avg = T_avg_arr[n_i] = np.average(
                wire.record_dict["T_distribution"][-1])

            R_arr[n_i] = wire.resistance_total()
            x_arr[n_i] = wire.x_offset_beam

        # print("T_avg_arr: ", T_avg_arr)
        # print("T_max_arr: ", T_max_arr)
        # print("signal_arr: ", signal_arr)
        # print("x_arr: ", x_arr)


        x_list = np.array([np.concatenate((x_pos_seg, -1 * x_pos_seg[::-1])),
                  np.concatenate((x_pos_seg, -1 * x_pos_seg[::-1]))])


        signal_arr_norm_1 = [0.0216639499386916, 
                             0.06400510252472039, 0.1053120495242299, 
         0.1455846628698668, 0.22303108769945335, 0.260209306315826, 
         0.29636195737211446, 0.3314934324624902, 0.3987144288917815,
           0.4308166043914375, 0.46192281167714666, 0.5211797239090269,
             0.5493479545349781, 0.5765551496525174, 0.6028111285969292,
               0.652510228987278, 0.6759743261085402, 0.6985290436389583, 
               0.7409536239600266, 0.7608451610624817, 0.7798705689299693,
                 0.7980405003681781, 0.8318557396310409, 0.8475214304508306,
                   0.8623723390671673, 0.8896675428283924, 0.9021298623376048,
                     0.913813396881382, 0.9247262045852894, 0.9442697530284341,
        0.9529144383444618, 0.9608162550428692, 0.9744139623616798, 
        0.9801199485931034, 0.9851032386668377, 0.989367596091101, 
        0.9957519299061783, 0.9978767811933983, 0.9992924418760787, 
        1.0, 1.0,
          0.9992924418760787, 0.9978767811933983, 0.9957519299061783,
            0.989367596091101, 0.9851032386668377, 0.9801199485931034,
              0.9744139623616798, 0.9608162550428692, 0.9529144383444618, 
              0.9442697530284341, 0.9247262045852894, 0.913813396881382,
                0.9021298623376048, 0.8896675428283924, 0.8623723390671673,
                  0.8475214304508306, 0.8318557396310409, 0.7980405003681781,
                    0.7798705689299693, 0.7608451610624817, 0.7409536239600266,
         0.6985290436389583, 0.6759743261085402, 0.652510228987278, 
         0.6028111285969292, 0.5765551496525174, 0.5493479545349781,
           0.5211797239090269, 0.46192281167714666, 0.4308166043914375,
             0.3987144288917815, 0.3314934324624902, 0.29636195737211446,
               0.260209306315826, 0.22303108769945335, 0.1455846628698668,
                 0.1053120495242299, 0.06400510252472039, 0.0216639499386916]

        plot_signal_over_x_off(np.transpose([signal_arr_norm,
                                             signal_arr_norm_1]), 
                        np.transpose(x_list), 
                        plotname = f"signal_over_x_off_seg_compare",
                        power  = power)