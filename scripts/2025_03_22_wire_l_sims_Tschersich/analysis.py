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
# exp_list = np.linspace(16,17,num = 1)   # later normalized to per cm**2
# exp_list = np.linspace(17,18,num = 1)   # later normalized to per cm**2
# exp_list = np.array([0,16,17])
# beamshape_list = [10, 1.6]

# l_wire_list = ([0.1 * i for i in range(2,20)] 
#              + [2 + 0.2 * i for i in range(0,26)]
#                )

l_wire_list = ([0.1 * i for i in range(2,20)] 
             + [2 + 0.2 * i for i in range(0,26)]
             + [8.0 + 1 * i for i in range(0,8)]
               )
# #Choose every 5th for faster running
# l_wire_list = l_wire_list[::5]
l_wire_list = np.round(np.array(l_wire_list),decimals=1)


phi_exp_list = np.linspace(17,18,num = 1) 
beamshape_list = ["Tschersich"]

R_arr_full = np.zeros((len(beamshape_list),len(l_wire_list), 
                       len(phi_exp_list)))
signal_arr_full = np.zeros((len(beamshape_list),len(l_wire_list), 
                            len(phi_exp_list)))
i_current = 0
for n_bs, beamshape in enumerate(beamshape_list):
    # Initialize Arrays
    R_arr = np.zeros((len(l_wire_list), len(phi_exp_list)))
    T_max_arr = np.zeros((len(l_wire_list), len(phi_exp_list)))
    T_avg_arr = np.zeros((len(l_wire_list), len(phi_exp_list)))
    signal_arr = np.zeros((len(l_wire_list), len(phi_exp_list)))
    for n_lw, l_wire in enumerate(l_wire_list):
        for n_p, phi_exp in enumerate(phi_exp_list):
            try:
                run_name = "bs_{}_lw_{}_phi_{}_i_{}".format(
                    beamshape, l_wire, phi_exp,i_current)
                #TODO Include enumerates, output plots for every i_current
                wire = Wire()
                wire = wire.load(top_dir + "results\\" + run_name)
            except:
                run_name = "bs_{}_lw_{}_phi_{}_i_{}".format(
                    beamshape, int(l_wire), phi_exp,i_current)
                #TODO Include enumerates, output plots for every i_current
                wire = Wire()
                wire = wire.load(top_dir + "results\\" + run_name)
            #l_beam = wire.l_beam

            #Switch U_arr for R_arr

            # U_beam_off = wire.U_wire(0)
            # U_beam_on = wire.U_wire(-1)

            R_initial = wire.resistance_total(
                        wire.record_dict["T_distribution"][0])
            R_final = wire.resistance_total(
                        wire.record_dict["T_distribution"][-1])
            
            #HACK
            U_beam_off = R_initial
            U_beam_on = R_final
            
            U_delta = R_arr[n_lw, n_p] = U_beam_on - U_beam_off
            signal = signal_arr[n_lw, n_p] = U_delta / U_beam_off

            T_max = T_max_arr[n_lw, n_p] = np.amax(
                wire.record_dict["T_distribution"][-1])
            T_avg = T_avg_arr[n_lw, n_p] = np.average(
                wire.record_dict["T_distribution"][-1])
        # phi_list = 10**exp_list
    R_arr_full[n_bs] = R_arr
    signal_arr_full[n_bs] = signal_arr


    
if True:
    for n_p, phi_exp in enumerate(phi_exp_list):
        fig = plt.figure(0, figsize=(8,6.5))
        ax1=plt.gca()

        label_list = ["l_eff = 4"]
        for n_bs, beamshape in enumerate(beamshape_list):
            ax1.plot(l_wire_list, signal_arr_full[n_bs, :, n_p],
                     "-", label=r"$10^{17}$ atoms/($\rm cm^2 s$),"
                                + "\n {}".format(
                         #phi_exp,
                                                label_list[n_bs]))

        ax1.set_ylabel(r"Relative Signal $\Delta R / R_{\rm initial}$")
        ax1.set_xlabel(r"Wire Length [cm]")
        plt.grid(True)
        plt.legend(shadow=True, title = "Power Input")

        format_im = 'png' #'pdf' or png
        dpi = 300
        plt.savefig(plot_dir + "rel_signal_vs_l_wire_pow_{}".format(
                                            phi_exp)
                    + '.{}'.format(format_im),
                    format=format_im, dpi=dpi, bbox_inches="tight")
        ax1.cla()

if False:
    for n_p, phi_exp in enumerate(phi_exp_list):
        fig = plt.figure(0, figsize=(8,6.5))
        ax1=plt.gca()

        for n_bs, beamshape in enumerate(beamshape_list):
            ax1.plot(l_wire_list, R_arr_full[n_bs, :, n_p],
                     "-", label="{}".format(beamshape) + r"$cm$")

        ax1.set_ylabel(r"$\Delta R$ [$\Omega$]")
        ax1.set_xlabel(r"wire length [cm]")
        plt.grid(True)
        plt.legend(shadow=True, title = "Beamspot Diameter")

        format_im = 'png' #'pdf' or png
        dpi = 300
        plt.savefig(plot_dir + "R_vs_l_wire_pow_{}".format(phi_exp)
                    + '.{}'.format(format_im),
                    format=format_im, dpi=dpi,
                    bbox_inches="tight")
        ax1.cla()

if True:
    #Custom labels
    for n_p, phi_exp in enumerate(phi_exp_list):
        fig = plt.figure(0, figsize=(8,6))
        ax1=plt.gca()

        label_list = ["l_eff = 4"]
        for n_bs, beamshape in enumerate(beamshape_list):
            ax1.plot(l_wire_list, R_arr_full[n_bs, :, n_p],
                     "-", label="{}, {}".format(phi_exp,
                                                label_list[n_bs]) )

        ax1.set_ylabel(r"$\Delta R$ [$\Omega$]")
        ax1.set_xlabel(r"Wire Length [cm]")
        plt.grid(True)
        plt.legend(shadow=True, title = "Power Input")

        format_im = 'png' #'pdf' or png
        dpi = 300
        plt.savefig(plot_dir + "R_vs_l_wire_pow_{}".format(phi_exp)
                    + '.{}'.format(format_im),
                    format=format_im, dpi=dpi,
                    bbox_inches="tight")
        ax1.cla()