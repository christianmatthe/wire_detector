import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as mpatches
import os
import dill
import sys

top_dir = os.path.dirname(os.path.abspath(__file__)) + "\\"
sys.path.append(top_dir + "..\\")
from wire_detector import Wire
from time import time

plot_dir = top_dir + "analysis/"
os.makedirs(plot_dir, exist_ok=True)
heat_flow_dir = top_dir + "heat_flow/"
os.makedirs(heat_flow_dir, exist_ok=True)


l_wire_list = ([2]
               )
T_bkgd_list = np.array([0,0.1, 1,2,10])+ 293.15

emissivity_list = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
i_current = 1




# Initialize Arrays
U_arr = np.zeros((len(emissivity_list), len(T_bkgd_list)))
T_avg_start_arr = np.zeros((len(emissivity_list), len(T_bkgd_list)))
T_avg_arr = np.zeros((len(emissivity_list), len(T_bkgd_list)))
signal_arr = np.zeros((len(emissivity_list), len(T_bkgd_list)))
for n_d, eps in enumerate(emissivity_list):
    for n_p, T_bkgd in enumerate(T_bkgd_list):
        run_name = "T_bkgd_{:.2f}_eps_{:.2f}_i_{}".format(
        T_bkgd, eps, i_current)
        #TODO Include enumerates, output plots for every i_current
        wire = Wire()
        wire = wire.load(top_dir + "results\\" + run_name)
        print("alive. loading: ", "n_d", n_d, "eps", eps, "n_p", n_p)
        l_beam = wire.l_beam

        # U_beam_off = wire.U_wire(0)
        # U_beam_on = wire.U_wire(-1)

        R_initial = wire.resistance_total(
                    wire.record_dict["T_distribution"][0])
        R_final = wire.resistance_total(
                    wire.record_dict["T_distribution"][-1])
        
        #HACK
        U_beam_off = R_initial
        U_beam_on = R_final
        
        U_delta = U_arr[n_d, n_p] = U_beam_on - U_beam_off
        signal = signal_arr[n_d, n_p] = U_delta / U_beam_off

        #HACK
        R_arr = U_arr

        T_avg_start = T_avg_start_arr[n_d, n_p] = np.average(
            wire.record_dict["T_distribution"][0])
        T_avg = T_avg_arr[n_d, n_p] = np.average(
            wire.record_dict["T_distribution"][-1])

        # if True:
        #     # Calculate endstate of heat flow
        #     x_lst = [1000 * ((i + 0.5) * wire.l_segment - (wire.l_wire / 2))
        #             for i in range(wire.n_wire_elements)]
        #     wire.T_distribution = wire.record_dict["T_distribution"][-1]
        #     f_el_arr = wire.f_el()
        #     f_conduction_arr = wire.f_conduction() 
                                
        #     f_rad_arr = wire.f_rad() 
        #     f_beam_arr = wire.f_beam()

        #     # Plot endstate of heat flow
        #     fig = plt.figure(0, figsize=(8,6.5))
        #     ax1=plt.gca()

        #     ax1.plot(x_lst, f_el_arr, "-", label=r"$F_{el}$")
        #     ax1.plot(x_lst, f_conduction_arr, "-", label=r"$F_{conduction}$")
        #     ax1.plot(x_lst, f_rad_arr, "-", label=r"$F_{rad}$")
        #     ax1.plot(x_lst, f_beam_arr, "-", label=r"$F_{beam}$")

        #     ax1.set_ylabel("Heat Flow [W/m]")
        #     ax1.set_xlabel(r"Wire positon [mm]")
        #     plt.grid(True)
        #     plt.legend(shadow=True)
        #     format_im = 'png' #'pdf' or png
        #     dpi = 300
        #     plt.savefig(heat_flow_dir + "heat_flow_d_{}_i_{}_phi_{}".format(
        #                 d, i_current, phi_exp) + '.{}'.format(format_im),
        #                 format=format_im, dpi=dpi)
        #     ax1.cla()


if True:
    # Plot Delta T_avg over emissivity

    fig = plt.figure(0, figsize=(8,6.5))
    ax1=plt.gca()

    DTs = [T_avg_arr[i][2] - T_avg_start_arr[i][2] 
           for i, eps in enumerate(emissivity_list)
                                                                     ]
    ax1.plot(emissivity_list, DTs, ".", 
                    label= r"$\Delta T_{board} =$ 1K", markersize = 10,
                    linestyle = "--")
    ax1.set_ylabel(r"$\kappa = \Delta T_{avg}/\Delta T_{board}$")
    ax1.set_xlabel(r"Emissivity $\epsilon$")
    plt.grid(True)
    plt.legend(shadow=True)


    format_im = 'png' #'pdf' or png
    dpi = 300
    plt.savefig(plot_dir + "deltaT_vs_eps".format(i_current) 
                + '.{}'.format(format_im),
                format=format_im, dpi=dpi, bbox_inches = "tight")
    ax1.cla()

