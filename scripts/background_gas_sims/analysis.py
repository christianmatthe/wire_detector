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


l_wire_list = [#5,
                2.7
               ] # in cm
exp_list = [14,16,17,18]
T_cracker_list = [ 2400
                #    ,2200,2000,1800,
                #    1000,500,
                #    300,0
                  ]


f_arr_full = np.zeros((len(l_wire_list), len(exp_list), len(T_cracker_list)
                       , 6  ))
for n_lw, l_wire in enumerate(l_wire_list):
    for n_phi, phi_exp in enumerate(exp_list):
        for n_T, T_cracker in enumerate(T_cracker_list):
            run_name = "lw_{}_phi_{}_Tc_{}".format(l_wire,phi_exp,
                            T_cracker)
            wire = Wire()
            wire = wire.load(top_dir + "results\\" + run_name)
            wire.plot_heat_flow(top_dir + "plots\\" 
                                + "heat_flow/log_{}".format(run_name)
                                , log_y = True)
            i = wire.n_wire_elements // 2
            elem = [wire.f_el(i), wire.f_conduction(i), wire.f_rad(i)
                    , wire.f_beam(i), wire.f_beam_gas(i), wire.f_bb(i)]
            f_arr_full[n_lw, n_phi, n_T] = elem

# Plot scaling of Heat flow with temperature with various 
if True:
    for n_lw, l_wire in enumerate(l_wire_list):
        for n_phi, phi_exp in enumerate(exp_list):
            fig = plt.figure(0, figsize=(8,6.5))
            ax1=plt.gca()
            label_list = [r"$F_{el}$", r"$F_{conduction}$", r"$F_{rad}$"
                          , r"$F_{beam}$", r"$F_{beam gas}$"
                          , r"$F_{bb cracker}$"]
            color_list = ["C0", "C1", "C2", "C3", "C4", "C5"]
            for n_f in [3,4,5]: # f_beam, f_beam_gas, f_bb
                T_lst = T_cracker_list
                f_lst = [f_arr_full[n_lw, n_phi, n_T, n_f]
                         for n_T in range(len(T_cracker_list))]
                ax1.plot(T_lst, f_lst,
                        "-", label=label_list[n_f], color=color_list[n_f])

            ax1.set_ylabel(r"Heat Flow [W/m]")
            ax1.set_xlabel(r"$T_{cracker}$ [K]")
            plt.grid(True)
            plt.legend(shadow=True)

            format_im = 'png' #'pdf' or png
            dpi = 300
            plt.savefig(plot_dir + "f_vs_T_phi_{}".format(phi_exp)
                        + '.{}'.format(format_im),
                        format=format_im, dpi=dpi)
            plt.yscale('log')
            plt.xscale('log')
            plt.savefig(plot_dir + "log_f_vs_T_phi_{}".format(phi_exp)
            + '.{}'.format(format_im),
            format=format_im, dpi=dpi)

            ax1.cla()

# U_arr_full = np.zeros((len(l_wire_list), len(i_current_list) ))
# signal_arr_full = np.zeros((len(l_wire_list), len(i_current_list)))
# for n_lw, l_wire in enumerate(l_wire_list):
#     # Initialize Arrays
#     U_arr = np.zeros(( len(i_current_list)))
#     T_max_arr = np.zeros(( len(i_current_list)))
#     T_avg_arr = np.zeros(( len(i_current_list)))
#     signal_arr = np.zeros(( len(i_current_list)))
#     R_arr = np.zeros(( len(i_current_list)))
#     for n_i, i_current in enumerate(i_current_list):
#         run_name = "lw_{}_i_{}".format(l_wire,i_current)
#         #TODO Include enumerates, output plots for every i_current
#         wire = Wire()
#         wire = wire.load(top_dir + "results\\" + run_name)
#         #l_beam = wire.l_beam

#         U_beam_off = wire.U_wire(0)
#         U_beam_on = wire.U_wire(-1)
        
#         U_delta = U_arr[n_i] = U_beam_on - U_beam_off
#         signal = signal_arr[n_i] = U_delta / U_beam_off

#         T_max = T_max_arr[n_i] = np.amax(
#             wire.record_dict["T_distribution"][-1])
#         T_avg = T_avg_arr[n_i] = np.average(
#             wire.record_dict["T_distribution"][-1])

#         R_arr[n_i] = wire.resistance_total()
#     U_arr_full[n_lw] = U_arr
#     signal_arr_full[n_lw] = signal_arr


# if True:
#         fig = plt.figure(0, figsize=(8,6.5))
#         ax1=plt.gca()
#         for n_lw, l_wire in enumerate(l_wire_list):
#             ax1.plot(i_current_list, R_arr,
#                     "-", label="{}".format(l_wire) + r"$cm$")

#         ax1.set_ylabel(r"Resistance [$\Omega$]")
#         ax1.set_xlabel(r"Current [mA]")
#         plt.grid(True)
#         plt.legend(shadow=True, title = "Wire Length")

#         format_im = 'png' #'pdf' or png
#         dpi = 300
#         plt.savefig(plot_dir + "R_vs_I"
#                     + '.{}'.format(format_im),
#                     format=format_im, dpi=dpi)
#         ax1.cla()

# if True:
#         fig = plt.figure(0, figsize=(8,6.5))
#         ax1=plt.gca()
#         for n_lw, l_wire in enumerate(l_wire_list):
#             ax1.plot(i_current_list, R_arr,
#                     "-", label="{}".format(l_wire) + r"$cm$")

#         ax1.set_ylabel(r"Resistance [$\Omega$]")
#         ax1.set_xlabel(r"Current [mA]")
#         plt.grid(True, which = "minor")
#         plt.legend(shadow=True, title = "Wire Length")

#         plt.yscale('log')
#         plt.xscale('log')

#         format_im = 'png' #'pdf' or png
#         dpi = 300
#         plt.savefig(plot_dir + "log_R_vs_I"
#                     + '.{}'.format(format_im),
#                     format=format_im, dpi=dpi)
#         ax1.cla()

# if True:
#     # Plot in air measurement 
#     i_list = [1,2,3,4,5,6,7,8,9,10]
#     U_list = [230,331,420,520,618,713,817, 925, 1026, 1137]
#     R_list = [(U_list[i]-134)/i_list[i] for i in range(len(i_list))]
#     fig = plt.figure(0, figsize=(8,6.5))
#     ax1=plt.gca()
#     for n_lw, l_wire in enumerate(l_wire_list):
#         ax1.plot(i_list, R_list,
#                 "-", label="{}".format(l_wire) + r"$cm$")

#     ax1.set_ylabel(r"Resistance [$\Omega$]")
#     ax1.set_xlabel(r"Current [mA]")
#     plt.grid(True)
#     plt.legend(shadow=True, title = "Wire Length")

#     format_im = 'png' #'pdf' or png
#     dpi = 300
#     plt.savefig(plot_dir + "in_air_real_R_vs_I"
#                 + '.{}'.format(format_im),
#                 format=format_im, dpi=dpi)
#     ax1.cla()

# if True:
#     fig = plt.figure(0, figsize=(8,6.5))
#     ax1=plt.gca()
#     colors = ["C0", "C1"]
#     for n_lw, l_wire in enumerate(l_wire_list):
#         ax1.plot(x_offset_list, U_arr_full[0, n_lw, 0, :]*1000,
#                     "-", label="{}".format(l_wire) + r"$cm$",
#                     color = colors[n_lw])
#         ax1.axvline((l_wire - 1.6)/2, color = colors[n_lw], ls = ":")

#     ax1.set_ylabel(r"$\Delta U$ [mV]")
#     ax1.set_xlabel(r"x offset [cm]")
#     plt.grid(True)
#     plt.legend(shadow=True, title = "Wire Length:")

#     format_im = 'png' #'pdf' or png
#     dpi = 300
#     plt.savefig(plot_dir + "U_vs_x_off"
#                 + '.{}'.format(format_im),
#                 format=format_im, dpi=dpi)
#     ax1.cla()

# if True:
#     fig = plt.figure(0, figsize=(8,6.5))
#     ax1=plt.gca()
#     colors = ["C0", "C1"]
#     for n_lw, l_wire in enumerate(l_wire_list):
#         ax1.plot(x_offset_list, signal_arr_full[0, n_lw, 0, :],
#                     "-", label="{}".format(l_wire) + r"$cm$",
#                     color = colors[n_lw])
#         ax1.axvline((l_wire - 1.6)/2, color = colors[n_lw], ls = ":")

#     ax1.set_ylabel(r"relative signal")
#     ax1.set_xlabel(r"x offset [cm]")
#     plt.grid(True)
#     plt.legend(shadow=True, title = "Wire Length:")

#     format_im = 'png' #'pdf' or png
#     dpi = 300
#     plt.savefig(plot_dir + "rel_signal_vs_x_off"
#                 + '.{}'.format(format_im),
#                 format=format_im, dpi=dpi)
#     ax1.cla()
