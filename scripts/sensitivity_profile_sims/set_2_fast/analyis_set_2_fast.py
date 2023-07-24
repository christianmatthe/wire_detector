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

if __name__ == "__main__":
    top_dir = os.path.dirname(os.path.abspath(__file__)) + os.sep

    plot_dir = top_dir + "analysis_2023_06_12/".format()
    os.makedirs(plot_dir, exist_ok=True)
    heat_flow_dir = top_dir + "heat_flow/"

    ##### Constants
    E_recombination = Wire().E_recombination
    segment_length = Wire(n_wire_elements = 100,l_wire=2*10**-2).l_segment
    emissivity = 0.08
    pressure = 0
    d = 5
    i_current = 100.0
    l_wire = 2
    ####

    ### scan lists
    x_pos_list = np.linspace(-1,1, 20, endpoint= False) + segment_length/2
    # print(x_pos_list)
    # power_watts_list = [100e-6, 1e-6]
    ###

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
        #wire = wire.load(top_dir + "0.02mbar_air\\" + "results\\" + run_name)
        wire = wire.load(top_dir + "{}muW_x_off_{}\\".format(
                        i_current, n_i) 
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
    print("signal_arr: ", signal_arr)
    print("x_arr: ", x_arr)

    #### Plot
    def plot_signal_over_x_off(signal_arr, x_arr, 
                               plotname = "signal_over_x_off"):
        fig = plt.figure(0, figsize=(8,6.5))
        ax1=plt.gca()
        ax1.plot(x_arr * 1e3, signal_arr, label = "100µW")

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
        
    plot_signal_over_x_off(signal_arr, x_arr, 
                               plotname = "signal_over_x_off")