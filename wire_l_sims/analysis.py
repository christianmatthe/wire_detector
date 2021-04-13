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


#d_wire_list = [5]
#i_current_list =  [1]
exp_list = np.linspace(14,18,num = 5)  # later normalized to per cm**2
l_beam_list = [10, 1.6]
l_wire_list = [5,4,3,
               2.5,2.4,2.3,2.2,2.1,
               2,1.9,1.7,1.5,1,
               0.5,0.2,0.1] # in cm

U_arr_full = np.zeros((len(l_beam_list),len(l_wire_list), len(exp_list)))
signal_arr_full = np.zeros((len(l_beam_list),len(l_wire_list), len(exp_list)))
for n_lb, l_beam in enumerate(l_beam_list):
    # Initialize Arrays
    U_arr = np.zeros((len(l_wire_list), len(exp_list)))
    T_max_arr = np.zeros((len(l_wire_list), len(exp_list)))
    T_avg_arr = np.zeros((len(l_wire_list), len(exp_list)))
    signal_arr = np.zeros((len(l_wire_list), len(exp_list)))
    for n_lw, l_wire in enumerate(l_wire_list):
        for n_p, phi_exp in enumerate(exp_list):
            run_name = "lb_{}_lw_{}_phi_{}".format(
                l_beam, l_wire, phi_exp)
            #TODO Include enumerates, output plots for every i_current
            wire = Wire()
            wire = wire.load(top_dir + "results\\" + run_name)
            #l_beam = wire.l_beam

            U_beam_off = wire.U_wire(0)
            U_beam_on = wire.U_wire(-1)
            
            U_delta = U_arr[n_lw, n_p] = U_beam_on - U_beam_off
            signal = signal_arr[n_lw, n_p] = U_delta / U_beam_off

            T_max = T_max_arr[n_lw, n_p] = np.amax(
                wire.record_dict["T_distribution"][-1])
            T_avg = T_avg_arr[n_lw, n_p] = np.average(
                wire.record_dict["T_distribution"][-1])
        phi_list = 10**exp_list
    U_arr_full[n_lb] = U_arr
    signal_arr_full[n_lb] = signal_arr


    if True:
        # Plot delta U vs Phi in atoms/s
        A_beam = np.pi * ((l_beam)/2)**2 # in cm**2

        fig = plt.figure(0, figsize=(8,6.5))
        ax1=plt.gca()

        for n_lw, l_wire in enumerate(l_wire_list):
            ax1.loglog(A_beam * phi_list, U_arr[n_lw]*1000, "-", 
                        label="{}".format(l_wire) 
                        + r"$cm$", basex=10)
        ax1.set_ylabel(r"$\Delta U$ [mV]")
        ax1.set_xlabel(r"$\Phi_{beam}$ [Atoms/s]")
        plt.grid(True)
        plt.legend(shadow=True, title = "Wire Length")


        format_im = 'png' #'pdf' or png
        dpi = 300
        plt.savefig(plot_dir + "deltaU_compare_lb_{}".format(l_beam) 
                    + '.{}'.format(format_im),
                    format=format_im, dpi=dpi)
        ax1.cla()
    
    if True:
        # Plot delta U vs Phi in atoms/(s * cm**2)
        A_beam = np.pi * ((l_beam)/2)**2


        fig = plt.figure(0, figsize=(8,6.5))
        ax1=plt.gca()

        for n_lw, l_wire in enumerate(l_wire_list):
            ax1.loglog(phi_list, U_arr[n_lw]*1000, "-", 
                       label="{}".format(l_wire) 
                       + r"$cm$", basex=10)
        ax1.set_ylabel(r"$\Delta U$ [mV]")
        ax1.set_xlabel(r"$\Phi_{beam}$ [Atoms/$cm^2 s$]")
        plt.grid(True)
        plt.legend(shadow=True, title = "Wire Length")


        format_im = 'png' #'pdf' or png
        dpi = 300
        plt.savefig(plot_dir + "deltaU_compare_area_norm_lb_{}".format(l_beam)
                    + '.{}'.format(format_im),
                    format=format_im, dpi=dpi)
        ax1.cla()
    
    if True:
        # Plot relative signal vs Phi in atoms/(s * cm**2)
        A_beam = np.pi * ((l_beam)/2)**2


        fig = plt.figure(0, figsize=(8,6.5))
        ax1=plt.gca()

        for n_lw, l_wire in enumerate(l_wire_list):
            ax1.loglog(phi_list, signal_arr[n_lw], "-", 
                       label="{}".format(l_wire) 
                       + r"$cm$", basex=10)
        ax1.set_ylabel(r"relative signal")
        ax1.set_xlabel(r"$\Phi_{beam}$ [Atoms/$cm^2 s$]")
        plt.grid(True)
        plt.legend(shadow=True, title = "Wire Length")


        format_im = 'png' #'pdf' or png
        dpi = 300
        plt.savefig(plot_dir + "rel_signal_area_norm_lb_{}".format(l_beam)
                    + '.{}'.format(format_im),
                    format=format_im, dpi=dpi)
        ax1.cla()
    
if True:
    for n_p, phi_exp in enumerate(exp_list):
        fig = plt.figure(0, figsize=(8,6.5))
        ax1=plt.gca()

        for n_lb, l_beam in enumerate(l_beam_list):
            ax1.plot(l_wire_list, signal_arr_full[n_lb, :, n_p],
                     "-", label="{}".format(l_beam) + r"$cm$")

        ax1.set_ylabel(r"relative signal")
        ax1.set_xlabel(r"wire length [cm]")
        plt.grid(True)
        plt.legend(shadow=True, title = "Beamspot Diameter")

        format_im = 'png' #'pdf' or png
        dpi = 300
        plt.savefig(plot_dir + "rel_signal_vs_l_wire_phi_{}".format(phi_exp)
                    + '.{}'.format(format_im),
                    format=format_im, dpi=dpi)
        ax1.cla()

if True:
    for n_p, phi_exp in enumerate(exp_list):
        fig = plt.figure(0, figsize=(8,6.5))
        ax1=plt.gca()

        for n_lb, l_beam in enumerate(l_beam_list):
            ax1.plot(l_wire_list, U_arr_full[n_lb, :, n_p]*1000,
                     "-", label="{}".format(l_beam) + r"$cm$")

        ax1.set_ylabel(r"$\Delta U$ [mV]")
        ax1.set_xlabel(r"wire length [cm]")
        plt.grid(True)
        plt.legend(shadow=True, title = "Beamspot Diameter")

        format_im = 'png' #'pdf' or png
        dpi = 300
        plt.savefig(plot_dir + "U_vs_l_wire_phi_{}".format(phi_exp)
                    + '.{}'.format(format_im),
                    format=format_im, dpi=dpi)
        ax1.cla()