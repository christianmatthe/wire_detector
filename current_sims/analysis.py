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
i_current_list = [1,2,3,4,5,6,7,8,9,10]
l_wire_list = [#5,
                2.7
               ] # in cm

U_arr_full = np.zeros((len(l_wire_list), len(i_current_list) ))
signal_arr_full = np.zeros((len(l_wire_list), len(i_current_list)))
for n_lw, l_wire in enumerate(l_wire_list):
    # Initialize Arrays
    U_arr = np.zeros(( len(i_current_list)))
    T_max_arr = np.zeros(( len(i_current_list)))
    T_avg_arr = np.zeros(( len(i_current_list)))
    signal_arr = np.zeros(( len(i_current_list)))
    R_arr = np.zeros(( len(i_current_list)))
    for n_i, i_current in enumerate(i_current_list):
        run_name = "lw_{}_i_{}".format(l_wire,i_current)
        #TODO Include enumerates, output plots for every i_current
        wire = Wire()
        wire = wire.load(top_dir + "results\\" + run_name)
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
    U_arr_full[n_lw] = U_arr
    signal_arr_full[n_lw] = signal_arr


if True:
        fig = plt.figure(0, figsize=(8,6.5))
        ax1=plt.gca()
        for n_lw, l_wire in enumerate(l_wire_list):
            ax1.plot(i_current_list, R_arr,
                    "-", label="{}".format(l_wire) + r"$cm$")

        ax1.set_ylabel(r"Resistance [$\Omega$]")
        ax1.set_xlabel(r"Current [mA]")
        plt.grid(True)
        plt.legend(shadow=True, title = "Wire Length")

        format_im = 'png' #'pdf' or png
        dpi = 300
        plt.savefig(plot_dir + "R_vs_I"
                    + '.{}'.format(format_im),
                    format=format_im, dpi=dpi)
        ax1.cla()

if True:
        fig = plt.figure(0, figsize=(8,6.5))
        ax1=plt.gca()
        for n_lw, l_wire in enumerate(l_wire_list):
            ax1.plot(i_current_list, R_arr,
                    "-", label="{}".format(l_wire) + r"$cm$")

        ax1.set_ylabel(r"Resistance [$\Omega$]")
        ax1.set_xlabel(r"Current [mA]")
        plt.grid(True, which = "minor")
        plt.legend(shadow=True, title = "Wire Length")

        plt.yscale('log')
        plt.xscale('log')

        format_im = 'png' #'pdf' or png
        dpi = 300
        plt.savefig(plot_dir + "log_R_vs_I"
                    + '.{}'.format(format_im),
                    format=format_im, dpi=dpi)
        ax1.cla()

if True:
    # Plot in air measurement 
    i_list = [1,2,3,4,5,6,7,8,9,10]
    U_list = [230,331,420,520,618,713,817, 925, 1026, 1137]
    R_list = [(U_list[i]-134)/i_list[i] for i in range(len(i_list))]
    fig = plt.figure(0, figsize=(8,6.5))
    ax1=plt.gca()
    for n_lw, l_wire in enumerate(l_wire_list):
        ax1.plot(i_list, R_list,
                "-", label="{}".format(l_wire) + r"$cm$")

    ax1.set_ylabel(r"Resistance [$\Omega$]")
    ax1.set_xlabel(r"Current [mA]")
    plt.grid(True)
    plt.legend(shadow=True, title = "Wire Length")

    format_im = 'png' #'pdf' or png
    dpi = 300
    plt.savefig(plot_dir + "in_air_real_R_vs_I"
                + '.{}'.format(format_im),
                format=format_im, dpi=dpi)
    ax1.cla()

if True:
    # Plot in air measurement 
    i_list = [1,2,3,4,5,6,7,8,9,10]
    U_list = [230,331,420,520,618,713,817, 925, 1026, 1137]
    R_list = [(U_list[i]-134)/i_list[i] for i in range(len(i_list))]
    fig = plt.figure(0, figsize=(8,6.5))
    ax1=plt.gca()
    for n_lw, l_wire in enumerate(l_wire_list):
        ax1.plot(i_list, R_list,
                "-", label="{}".format(l_wire) + r"$cm$")

    ax1.set_ylabel(r"Resistance [$\Omega$]")
    ax1.set_xlabel(r"Current [mA]")
    plt.grid(True)
    plt.legend(shadow=True, title = "Wire Length")

    format_im = 'png' #'pdf' or png
    dpi = 300
    plt.savefig(plot_dir + "in_air_real_R_vs_I"
                + '.{}'.format(format_im),
                format=format_im, dpi=dpi)
    ax1.cla()

if True:
    # Proper 4 wire measurement
    i_list = [1.49, 2.51, 3.52, 4.43, 5.45, 6.46, 7.36, 8.38, 9.41, 10.31,
              11.34, 12.35, 13.38, 14.39, 15.32, 16.30, 17.35, 21.2, 26.1,
              30.9, 32.0

              , 32.0, 35.0, 37.0, 39.0, 40.7, 42.4, 44.0, 45.8
              ]
    U_list = [143, 239, 336, 432, 522, 622, 711, 818, 921, 1025, 1135, 1250,
              1372, 1490, 1605, 1740, 1870, 2450, 3330, 4500, 4900

              , 4910, 5900, 6890, 7890, 8880, 9880, 11870, 13860
              ]
    i_err = [0.02, 0.02,0.02,0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02
             , 0.02,0.02, 0.03, 0.03, 0.03, 0.03, 0.1, 0.1, 0.2, 0.3

             , 0.5, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8 
             ]
    U_err = [2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 4, 4, 4, 10, 10, 10, 15, 30, 100
             , 100, 10
             
             , 10, 10, 10, 10, 10, 10, 10, 10
             ]
    R_list = [(U_list[i])/i_list[i] for i in range(len(i_list))]
    R_err = [np.sqrt(((U_err[i])/i_list[i])**2 
                      + (i_err[i]*(U_list[i])/i_list[i]**2)**2)
             for i in range(len(i_list))]
    fig = plt.figure(0, figsize=(8,6.5))
    ax1=plt.gca()
    for n_lw, l_wire in enumerate(l_wire_list):
        # ax1.plot(i_list, R_list,
        #         "-", label="{}".format(l_wire) + r"$cm$")
        ax1.errorbar(i_list, R_list, R_err, i_err,
                ".", label="{}".format(l_wire) + r"$cm$")

    ax1.set_ylabel(r"Resistance [$\Omega$]")
    ax1.set_xlabel(r"Current [mA]")
    plt.grid(True)
    plt.legend(shadow=True, title = "Wire Length")

    format_im = 'png' #'pdf' or png
    dpi = 300
    plt.savefig(plot_dir + "in_air_real_R_vs_I_4wire_err"
                + '.{}'.format(format_im),
                format=format_im, dpi=dpi)
    ax1.cla()

    #Power vs Temp plot
    P_list = [10**-3 * (i_list[i])**2* R_list[i] for i in range(len(i_list))]
    P_err = [10**-3 * np.sqrt((i_err[i] * 2* i_list[i] * R_list[i])**2 
                      + (R_err[i]*i_list[i]**2)**2)
             for i in range(len(i_list))]
    alpha = 4.7*10**-3
    dT_list = [(1/alpha)*(R_list[i]/R_list[0] - 1) for i in range(len(i_list))]
    dT_err = [(1/alpha)*np.sqrt((R_err[i]/R_list[0])**2 
                    + (R_err[0]*R_list[i]/(R_list[0]**2))**2)
            for i in range(len(i_list))]
    # dT_err = [5
    #         for i in range(len(i_list))]
    T_list  = [25 + dT_list[i] for i in range(len(i_list))]
    
    fig = plt.figure(0, figsize=(8,6.5))
    ax1=plt.gca()
    for n_lw, l_wire in enumerate(l_wire_list):
        # ax1.plot(i_list, R_list,
        #         "-", label="{}".format(l_wire) + r"$cm$")
        ax1.errorbar(P_list, T_list, dT_err, P_err,
                ".", label="{}".format(l_wire) + r"$cm$")

    ax1.set_ylabel(r"Average Temperature [°C]")
    ax1.set_xlabel(r"Power [mW]")
    plt.grid(True)
    plt.legend(shadow=True, title = "Wire Length")

    format_im = 'png' #'pdf' or png
    dpi = 300
    plt.savefig(plot_dir + "in_air_real_P_vs_T"
                + '.{}'.format(format_im),
                format=format_im, dpi=dpi)
    ax1.cla()
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
