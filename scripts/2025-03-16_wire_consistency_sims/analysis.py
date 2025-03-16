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


# d_wire_list = [0.6,5,10,20]
# i_current_list = [0.1, 1]
# exp_list = np.linspace(14,20,num = 25)  # later normalized to per cm**2
d = 5
i_current = 1
# n_seg_lst = [10,20,50,100,200]
n_seg_lst = [6,10,20,30, 40, 50,60, 80, 100, 120, 140, 160, 180, 200, 220, 240]

# Initialize Arrays
U_arr = np.zeros(len(n_seg_lst))
T_max_arr = np.zeros(len(n_seg_lst))
T_avg_arr = np.zeros(len(n_seg_lst))
signal_arr = np.zeros(len(n_seg_lst))

R_wire_arr = np.zeros(len(n_seg_lst))
    

for n, n_seg in enumerate(n_seg_lst):
    run_name = "d_{}_i_{}_nseg_{}".format(d, i_current, n_seg)
    #TODO Include enumerates, output plots for every i_current
    wire = Wire()
    wire = wire.load(top_dir + "results\\" + run_name)
    l_beam = wire.l_beam

    # U_beam_off = wire.U_wire(0)
    # U_beam_on = wire.U_wire(-1)
    
    # U_delta = U_arr[n_d, n_p] = U_beam_on - U_beam_off
    # signal = signal_arr[n_d, n_p] = U_delta / U_beam_off

    # T_max = T_max_arr[n_d, n_p] = np.amax(
    #     wire.record_dict["T_distribution"][-1])
    # T_avg = T_avg_arr[n_d, n_p] = np.average(
    #     wire.record_dict["T_distribution"][-1])
    
    R_wire = wire.resistance_total(wire.record_dict["T_distribution"][-1])

    R_wire_arr[n] = R_wire


if True:
    # Plot R_wire vs n_seg

    fig = plt.figure(0, figsize=(8,6.5))
    ax1=plt.gca()

    ax1.plot(n_seg_lst, R_wire_arr, "x",
             ms = 10, 
                # label="{}".format(d) 
                # + r"$\mu m$"
                )
    ax1.set_ylabel(r"$R$ [$\Omega$]")
    ax1.set_xlabel(r"$n_{\rm seg}$ ")
    plt.grid(True)
    plt.legend(shadow=True)
    plt.tight_layout()


    format_im = 'png' #'pdf' or png
    dpi = 300
    plt.savefig(plot_dir + "R_vs_nseg".format() 
                + '.{}'.format(format_im),
                format=format_im, dpi=dpi)
    ax1.cla()

if True:
    # Plot R_wire vs n_seg

    fig = plt.figure(0, figsize=(8,6.5))
    ax1=plt.gca()

    ax1.plot(n_seg_lst, R_wire_arr - R_wire_arr[-1] , ".", 
             ms = 10,
                # label="{}".format(d) 
                # + r"$\mu m$"
                )
    ax1.set_ylabel(r"$R_{diff}$ [$\Omega$]")
    ax1.set_xlabel(r"$n_{\rm seg}$ ")
    plt.grid(True)
    plt.legend(shadow=True)
    plt.tight_layout()


    format_im = 'png' #'pdf' or png
    dpi = 300
    plt.savefig(plot_dir + "R_diff_vs_nseg".format() 
                + '.{}'.format(format_im),
                format=format_im, dpi=dpi)
    ax1.cla()

if True:
    # Plot R_wire vs n_seg

    fig = plt.figure(0, figsize=(8,6.5))
    ax1=plt.gca()

    ax1.plot(n_seg_lst, (R_wire_arr - R_wire_arr[-1])/R_wire_arr[-1] , ".", 
             ms = 10,
                # label="{}".format(d) 
                # + r"$\mu m$"
                )
    ax1.set_ylabel(r"$R_{diff}$ [$\Omega$ / $\Omega$ ]")
    ax1.set_xlabel(r"$n_{\rm seg}$ ")
    plt.grid(True)
    plt.legend(shadow=True)
    plt.tight_layout()


    format_im = 'png' #'pdf' or png
    dpi = 300
    plt.savefig(plot_dir + "R_prop_vs_nseg".format() 
                + '.{}'.format(format_im),
                format=format_im, dpi=dpi)
    

    ax1.set_yscale("log")
    plt.savefig(plot_dir + "R_prop_vs_nseg_log".format() 
                + '.{}'.format(format_im),
                format=format_im, dpi=dpi)
    ax1.cla()
    ax1.cla()

