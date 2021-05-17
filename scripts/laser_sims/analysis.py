import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as mpatches
import os
import dill
import sys

top_dir = os.path.dirname(os.path.abspath(__file__)) + "\\"
sys.path.append(top_dir + "..\\")
from Wire_detector import Wire
from time import time

plot_dir = top_dir + "analysis/"
os.makedirs(plot_dir, exist_ok=True)
heat_flow_dir = top_dir + "heat_flow/"

l_beam_list = [0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0
               ,2.2, 2.4, 2.6, 2.8]  # in cm


# Implement comparison of integrated heat flows over beam spot size
# Generate array:
func_list = ["f_el", "f_rad", "f_conduction", "f_beam",
             "f_beam_gas", "f_bb", "f_background_gas", "f_laser"] 

power_arr_full = np.zeros((len(func_list), len(l_beam_list)))
for n_lb, l_beam in enumerate(l_beam_list):
    run_name = "lb_{}".format(l_beam)
    wire = Wire().load(top_dir + "results\\" + run_name)
    for n_func,func in enumerate(func_list):
        power = wire.integrate_f(getattr(wire, func))
        power_arr_full[n_func, n_lb] = power

if True:
    fig = plt.figure(0, figsize=(8,6.5))
    ax1=plt.gca()
    label_list = [r"$P_{el}$", r"$P_{conduction}$", r"$P_{rad}$"
                     , r"$P_{beam}$", r"$P_{beam gas}$"
                     , r"$P_{bb cracker}$", r"$P_{background gas}$"
                     , r"$P_{laser}$"]
    # color_list = ["C0", "C1", "C2", "C3", "C4", "C5"]
    for n_func,func in enumerate(func_list):
        x_lst = l_beam_list
        p_lst = power_arr_full[n_func]
        # p_lst = (power_arr_full[n_func]/ power_arr_full[-1]
        #         * np.average(power_arr_full[-1]))
        ax1.plot(x_lst, p_lst,
                "-", label=label_list[n_func]
                #,color=color_list[n_f]
                 )

    ax1.set_ylabel(r"Power [W]")
    ax1.set_xlabel(r"Beam width [cm]")
    plt.grid(True)
    plt.legend(shadow=True)

    format_im = 'png' #'pdf' or png
    dpi = 300
    plt.savefig(plot_dir + "P_vs_lb"
                + '.{}'.format(format_im),
                format=format_im, dpi=dpi)
    plt.yscale('log')
    #plt.xscale('log')
    plt.savefig(plot_dir + "log_P_vs_lb"
    + '.{}'.format(format_im),
    format=format_im, dpi=dpi)

    ax1.cla()