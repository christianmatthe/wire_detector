import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as mpatches
import os
import dill
import sys
#import scipy as sp
from scipy.interpolate import interp1d

top_dir = os.path.dirname(os.path.abspath(__file__)) + "\\"
sys.path.append(top_dir + "..\\")
from Wire_detector import Wire
from time import time

#plot Options
import matplotlib as mpl
font = {#'family' : 'normal','weight' : 'bold',
        'size'   : 16
        #,'serif':['Helvetica']
        }
mpl.rc('font', **font)

plot_dir = top_dir + "analysis/"
os.makedirs(plot_dir, exist_ok=True)
heat_flow_dir = top_dir + "heat_flow/"


#d_wire_list = [5]
i_current_list = [1,2,3,4,5,6
                    #,7,8,9,10
                    ]
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


# Integrated power graph
if True:
    # Populate arrays
    func_list = ["f_el", "f_conduction", "f_rad"
                #, "f_beam", "f_beam_gas", "f_bb"
                , "f_background_gas"
             #, "f_laser"
              ] 
    power_arr_full = np.zeros((len(func_list), len(i_current_list)))
    resistance_arr = np.zeros( len(i_current_list))
    T_arr = np.zeros(len(i_current_list))
    for n_i, i_current in enumerate(i_current_list):
        run_name = "lw_{}_i_{}".format(2.7,i_current)
        wire = Wire().load(top_dir + "results\\" + run_name)
        for n_func,func in enumerate(func_list):
            power = wire.integrate_f(getattr(wire, func))
            power_arr_full[n_func, n_i] = power
            resistance = wire.resistance_total()
            resistance_arr[n_i] = resistance
            T_avg = np.average(wire.T_distribution)
            T_arr[n_i] = T_avg 
    # Generate interpolation objects
    f_cond_interpolated = interp1d(T_arr, power_arr_full[1]
                                   ,fill_value="extrapolate")
    f_rad_interpolated = interp1d(T_arr, power_arr_full[2]
                                  ,fill_value="extrapolate")
    ### Plots
    #  P over I
    fig = plt.figure(0, figsize=(8,6.5))
    ax1=plt.gca()
    label_list = [r"$P_{el}$", r"$P_{conduction}$", r"$P_{rad}$"
                     #, r"$P_{beam}$", r"$P_{beam gas}$"
                     #, r"$P_{bb cracker}$"
                     , r"$P_{background gas}$"
                     #, r"$P_{laser}$"
                     ]
    color_list = ["C0", "C1", "C2", "C6"]
    marker_list = ["-", "--", "--", "--"]
    for n_func,func in enumerate(func_list):
        x_lst = i_current_list
        p_lst = 10**3 * power_arr_full[n_func]
        # p_lst = (power_arr_full[n_func]/ power_arr_full[-1]
        #         * np.average(power_arr_full[-1]))
        ax1.plot(x_lst, p_lst,
                ls = marker_list[n_func], label=label_list[n_func]
                ,color=color_list[n_func]
                 )

    ax1.set_ylabel(r"Power [mW]")
    ax1.set_xlabel(r"Current [mA]")
    plt.grid(True)
    plt.legend(shadow=True)

    format_im = 'png' #'pdf' or png
    dpi = 300
    plt.savefig(plot_dir + "P_vs_i"
                + '.{}'.format(format_im),
                format=format_im, dpi=dpi)
    plt.yscale('log')
    #plt.xscale('log')
    plt.savefig(plot_dir + "log_P_vs_i"
    + '.{}'.format(format_im),
    format=format_im, dpi=dpi)

    ax1.cla()

    #  P over T
    fig = plt.figure(0, figsize=(8,6.5))
    ax1=plt.gca()
    label_list = [r"$P_{el}$", r"$P_{conduction}$", r"$P_{rad}$"
                     #, r"$P_{beam}$", r"$P_{beam gas}$"
                     #, r"$P_{bb cracker}$"
                     , r"$P_{background gas}$"
                     #, r"$P_{laser}$"
                     ]
    color_list = ["C0", "C1", "C2", "C6"]
    marker_list = ["-", "--", "--", "--"]
    for n_func,func in enumerate(func_list):
        x_lst = T_arr - 273.25
        p_lst = 10**3 * power_arr_full[n_func]
        # p_lst = (power_arr_full[n_func]/ power_arr_full[-1]
        #         * np.average(power_arr_full[-1]))
        ax1.plot(x_lst, p_lst,
                ls = marker_list[n_func], label=label_list[n_func]
                ,color=color_list[n_func]
                 )

    ax1.set_ylabel(r"Power [mW]")
    ax1.set_xlabel(r"T [°C]")
    plt.grid(True)
    plt.legend(shadow=True)

    format_im = 'png' #'pdf' or png
    dpi = 300
    plt.savefig(plot_dir + "P_vs_T"
                + '.{}'.format(format_im),
                format=format_im, dpi=dpi)
    plt.yscale('log')
    #plt.xscale('log')
    plt.savefig(plot_dir + "log_P_vs_T"
    + '.{}'.format(format_im),
    format=format_im, dpi=dpi)
    plt.yscale('log')
    plt.xscale('log')
    plt.savefig(plot_dir + "loglog_P_vs_T"
    + '.{}'.format(format_im),
    format=format_im, dpi=dpi)

    ax1.cla()

    # 0.66 mbar measurement
    # Proper 4 wire measurement
    i_list = [0.1047,0.5134,1.0275,1.5253,2.0075,2.8753,3.197,3.64,3.904,4.1273
             ,4.305,4.432,4.5306,4.5094,4.572,4.666,4.735,4.802,4.814,4.842
             ,4.863,4.882,1.4882,4.8869,0.1041,4.6448,4.957,5.025,5.084,5.149
             ,5.203,5.227,5.24,5.256
              ]
    U_list = [9.8 ,48.2 ,97.1 ,145.8 ,195.1 ,292.2 ,392.3 ,502 ,591 ,691 ,792 
              ,889 ,988 ,988 ,1086 ,1186 ,1186 ,1186 ,1186 ,1186 ,1186 ,1186 
              ,145 ,1186 ,9.65 ,987 ,1236 ,1285 ,1335 ,1383 ,1433 ,1433 ,1433 
              ,1433
              ]
    i_err = [0.0001 ,0.0001 ,0.0001 ,0.0002 ,0.0002 ,0.0002 ,0.001 ,0.001 
             ,0.001 ,0.002 ,0.003 ,0.003 ,0.003 ,0.003 ,0.003 ,0.003 
             ,0.003 ,0.003 ,0.003 ,0.003 ,0.003 ,0.003 ,0.0002 ,0.003 
             ,0.0002 ,0.001 ,0.003 ,0.003 ,0.003 ,0.003 ,0.003 ,0.003 
             ,0.003 ,0.003 
             ]
    U_err = [0.1 ,0.1 ,0.1 ,0.1 ,0.1 ,0.1 ,0.1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 
             ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,0.1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1
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
    plt.savefig(plot_dir + "0.6mbar_R_vs_I_4wire_err"
                + '.{}'.format(format_im),
                format=format_im, dpi=dpi)
    ax1.cla()

    #Power vs Temp plot
    P_list = [10**-6 * (i_list[i])**2* R_list[i] for i in range(len(i_list))]
    P_err = [10**-6 * np.sqrt((i_err[i] * 2* i_list[i] * R_list[i])**2 
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
        ax1.errorbar(np.array(P_list)*10**3, T_list, dT_err
                    , np.array(P_err)*10**3,
                ".", label="{}".format(l_wire) + r"$cm$")

    ax1.set_ylabel(r"Average Temperature [°C]")
    ax1.set_xlabel(r"Power [mW]")
    plt.grid(True)
    plt.legend(shadow=True, title = "Wire Length")

    format_im = 'png' #'pdf' or png
    dpi = 300
    plt.savefig(plot_dir + "0.6mbar_P_vs_T"
                + '.{}'.format(format_im),
                format=format_im, dpi=dpi)
    ax1.cla()

    #Synthetic P over T
    fig = plt.figure(0, figsize=(8,6.5))
    ax1=plt.gca()
    label_list = [r"$P_{el}$", r"$P_{conduction\,synth}$", r"$P_{rad\,synth}$"
                     #, r"$P_{beam}$", r"$P_{beam gas}$"
                     #, r"$P_{bb cracker}$"
                     , r"$P_{gas\,synth}$"
                     #, r"$P_{laser}$"
                     ]
    color_list = ["C0", "C1", "C2", "C6"]
    marker_list = ["-", "--", "--", "--"]
    P_el_arr = np.array(P_list)
    P_cond_synth_arr = np.array([f_cond_interpolated(T + 273.25) 
                                for T in T_list])
    P_rad_synth_arr = np.array([f_rad_interpolated(T + 273.25) 
                                for T in T_list])
    P_gas_synth_arr = np.array([P_el_arr[i] - P_cond_synth_arr[i] 
                                - P_rad_synth_arr[i]
                                for i in range(len(P_el_arr))])
    P_synth_arr = np.array([P_el_arr, P_cond_synth_arr, P_rad_synth_arr, 
                            P_gas_synth_arr])
    for n_func,func in enumerate(func_list):
        x_lst = np.array(T_list)
        p_lst = 10**3 * P_synth_arr[n_func]
        # p_lst = (power_arr_full[n_func]/ power_arr_full[-1]
        #         * np.average(power_arr_full[-1]))
        ax1.plot(x_lst, p_lst,
                marker = ".", label=label_list[n_func]
                ,color=color_list[n_func], linestyle="None"
                 )

    ax1.set_ylabel(r"Power [mW]")
    ax1.set_xlabel(r"T [°C]")
    plt.grid(True)
    plt.legend(shadow=True)

    format_im = 'png' #'pdf' or png
    dpi = 300
    plt.savefig(plot_dir + "P_vs_T_synth"
                + '.{}'.format(format_im),
                format=format_im, dpi=dpi)
    plt.yscale('log')
    #plt.xscale('log')
    plt.savefig(plot_dir + "log_P_vs_T_synth"
    + '.{}'.format(format_im),
    format=format_im, dpi=dpi)
    plt.yscale('log')
    plt.xscale('log')
    plt.savefig(plot_dir + "loglog_P_vs_T_synth"
    + '.{}'.format(format_im),
    format=format_im, dpi=dpi)

    ax1.cla()

    #Setteling at 1200 mV
    i_list_2 = [4.666, 4.735, 4.747, 4.755, 4.763, 4.772, 4.780, 4.802, 4.808
                , 4.818,4.814, 4.821, 4.831, 4.842, 4.859, 4.863, 4.882]
    t_list = np.array([28, 31, 32, 33, 34, 35, 36, 40, 41, 43, 45, 47, 49, 51, 53, 55
             , 62])
    ax1.errorbar(t_list- t_list[0], i_list_2, yerr = 0.003
                , marker = ".", linestyle="None"
                 )
    ax1.set_ylabel(r"Current [mA]")
    ax1.set_xlabel(r"time [min]")
    plt.grid(True)

    format_im = 'png' #'pdf' or png
    dpi = 300
    plt.savefig(plot_dir + "setteling_1200mV"
                + '.{}'.format(format_im),
                format=format_im, dpi=dpi)
    ax1.cla()
    

    # 1 bar measurement
    # Proper 4 wire measurement
    old_plot_dir = plot_dir
    plot_dir = old_plot_dir +"1bar/"
    os.makedirs(plot_dir, exist_ok=True)
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
    plt.savefig(plot_dir + "0.6mbar_R_vs_I_4wire_err"
                + '.{}'.format(format_im),
                format=format_im, dpi=dpi)
    ax1.cla()

    #Power vs Temp plot
    P_list = [10**-6 * (i_list[i])**2* R_list[i] for i in range(len(i_list))]
    P_err = [10**-6 * np.sqrt((i_err[i] * 2* i_list[i] * R_list[i])**2 
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
        ax1.errorbar(np.array(P_list)*10**3, T_list, dT_err
                    , np.array(P_err)*10**3,
                ".", label="{}".format(l_wire) + r"$cm$")

    ax1.set_ylabel(r"Average Temperature [°C]")
    ax1.set_xlabel(r"Power [mW]")
    plt.grid(True)
    plt.legend(shadow=True, title = "Wire Length")

    format_im = 'png' #'pdf' or png
    dpi = 300
    plt.savefig(plot_dir + "0.6mbar_P_vs_T"
                + '.{}'.format(format_im),
                format=format_im, dpi=dpi)
    ax1.cla()

    #Synthetic P over T
    fig = plt.figure(0, figsize=(8,6.5))
    ax1=plt.gca()
    label_list = [r"$P_{el}$", r"$P_{conduction\,synth}$", r"$P_{rad\,synth}$"
                     #, r"$P_{beam}$", r"$P_{beam gas}$"
                     #, r"$P_{bb cracker}$"
                     , r"$P_{gas\,synth}$"
                     #, r"$P_{laser}$"
                     ]
    color_list = ["C0", "C1", "C2", "C6"]
    marker_list = ["-", "--", "--", "--"]
    P_el_arr = np.array(P_list)
    P_cond_synth_arr = np.array([f_cond_interpolated(T + 273.25) 
                                for T in T_list])
    P_rad_synth_arr = np.array([f_rad_interpolated(T + 273.25) 
                                for T in T_list])
    P_gas_synth_arr = np.array([P_el_arr[i] - P_cond_synth_arr[i] 
                                - P_rad_synth_arr[i]
                                for i in range(len(P_el_arr))])
    P_synth_arr = np.array([P_el_arr, P_cond_synth_arr, P_rad_synth_arr, 
                            P_gas_synth_arr])
    for n_func,func in enumerate(func_list):
        x_lst = np.array(T_list)
        p_lst = 10**3 * P_synth_arr[n_func]
        # p_lst = (power_arr_full[n_func]/ power_arr_full[-1]
        #         * np.average(power_arr_full[-1]))
        ax1.plot(x_lst, p_lst,
                marker = ".", label=label_list[n_func]
                ,color=color_list[n_func], linestyle="None"
                 )

    ax1.set_ylabel(r"Power [mW]")
    ax1.set_xlabel(r"T [°C]")
    plt.grid(True)
    plt.legend(shadow=True)

    format_im = 'png' #'pdf' or png
    dpi = 300
    plt.savefig(plot_dir + "P_vs_T_synth"
                + '.{}'.format(format_im),
                format=format_im, dpi=dpi)
    plt.yscale('log')
    #plt.xscale('log')
    plt.savefig(plot_dir + "log_P_vs_T_synth"
    + '.{}'.format(format_im),
    format=format_im, dpi=dpi)
    plt.yscale('log')
    plt.xscale('log')
    plt.savefig(plot_dir + "loglog_P_vs_T_synth"
    + '.{}'.format(format_im),
    format=format_im, dpi=dpi)

    ax1.cla()
    plot_dir = old_plot_dir

##################################################################
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

if False:
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

if False:
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


if False:
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

