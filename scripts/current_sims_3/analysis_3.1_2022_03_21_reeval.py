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

top_dir = os.path.dirname(os.path.abspath(__file__)) + os.sep
# plot_dir = top_dir + "analysis_3
# emissivity = 0.67
emissivity = 0.25
plot_dir = top_dir + "analysis_3.1_em_{}_2022_03_21/".format(emissivity)
os.makedirs(plot_dir, exist_ok=True)
heat_flow_dir = top_dir + "heat_flow/"


#d_wire_list = [5]
i_current_list = [0.1,
                  0.5,0.7,1,1.2,1.5,
                  1.7, 1.9,
                  # 2,
                  2.0,
                  2.2, 2.4, 2.6, 2.8,
                  # 3,
                  3.0,
                  3.2, 3.4, 3.6, 3.8,
                  # 4,
                  4.0,
                  4.2, 4.4, 4.6, 4.8,
                  # 5,
                  5.2, 5.4, 5.6, 5.8,
                  # 6,7,8,9,10
                  6.0, 7.0, 8.0, 9.0, 10.0
                  ]
# i_current_list = [  0.5, 0.7, 
#                     1,
#                     1.2,1.5,
#                     2,3,4,5,6
#                     ,7
#                     #,8,9,10
#                     ]
l_wire_list = [#5,
                2.7
               ] # in cm

#pressure = 0.0005
pressure = 0.02

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
        #wire = wire.load(top_dir + "0.02mbar_air\\" + "results\\" + run_name)
        wire = wire.load(top_dir + "{}mbar_air_em_{}\\".format(
                         pressure, emissivity) 
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
    U_arr_full[n_lw] = U_arr
    signal_arr_full[n_lw] = signal_arr

print("T_avg_arr: ", T_avg_arr)
print("T_max_arr: ", T_max_arr)
def T_avg_to_T_max(T_avg):
    f_int = interp1d(T_avg_arr - 273.15, T_max_arr - 273.15
            #  ,
            #  kind = "cubic"
            ,fill_value="extrapolate"
                )
    return f_int(T_avg)

def T_max_to_T_avg(T_max):
    f_int = interp1d(T_max_arr - 273.15, T_avg_arr - 273.15
            #  ,
            #  kind = "cubic"
            ,fill_value="extrapolate"
                )
    return f_int(T_max)

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
        wire = wire.load(top_dir + "{}mbar_air_em_{}\\".format(
                         pressure, emissivity) 
                         + "results\\" + run_name)
        wire.gen_k_heat_cond_function()
        for n_func,func in enumerate(func_list):
            power = wire.integrate_f(getattr(wire, func))
            power_arr_full[n_func, n_i] = power
            resistance = wire.resistance_total()
            resistance_arr[n_i] = resistance
            T_avg = np.average(wire.T_distribution)
            T_arr[n_i] = T_avg 
    # Generate interpolation objects
    f_cond_interpolated = interp1d(T_arr, power_arr_full[1], kind = "cubic"
                                   ,fill_value="extrapolate")
    f_rad_interpolated = interp1d(T_arr, power_arr_full[2], kind = "cubic"
                                  ,fill_value="extrapolate")
    f_gas_interpolated = interp1d(T_arr, power_arr_full[3], kind = "cubic"
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
        x_lst = T_arr - 273.15
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

    # P vs delta T
    fig = plt.figure(0, figsize=(8,6.5))
    ax1=plt.gca()
    for n_func,func in enumerate(func_list):
        x_lst = T_arr - 298.15
        p_lst = 10**3 * power_arr_full[n_func]
        # p_lst = (power_arr_full[n_func]/ power_arr_full[-1]
        #         * np.average(power_arr_full[-1]))
        ax1.plot(x_lst, p_lst,
                ls = marker_list[n_func], label=label_list[n_func]
                ,color=color_list[n_func]
                 )

    ax1.set_ylabel(r"Power [mW]")
    ax1.set_xlabel(r"$\Delta$ T [K]")

    plt.grid(True)
    plt.legend(shadow=True)

    format_im = 'png' #'pdf' or png
    dpi = 300
    plt.savefig(plot_dir + "P_vs_deltaT"
                + '.{}'.format(format_im),
                format=format_im, dpi=dpi)
    plt.yscale('log')
    #plt.xscale('log')
    plt.savefig(plot_dir + "log_P_vs_deltaT"
    + '.{}'.format(format_im),
    format=format_im, dpi=dpi)
    plt.yscale('log')
    plt.xscale('log')
    plt.savefig(plot_dir + "loglog_P_vs_deltaT"
    + '.{}'.format(format_im),
    format=format_im, dpi=dpi)

    ax1.cla()


# Plot simulation  and data into same plot
def plot_compare_to_sim(data_frame, out_dir,
                        T_arr=T_arr,
                        power_arr_full=power_arr_full,
                        x_lim=[0,400],
                        y_lim=[-0.2,3],
                        y_lim_res=[-0.2,0.2]
                        ):

    # Observables to P_synth_arr
    U_list = data_frame["U meas (mV)"].values.tolist()
    U_err = data_frame["err U"].values.tolist()
    i_list = data_frame["I meas (mA)"].values.tolist()
    i_err = data_frame["err I"].values.tolist()

    R_list = [(U_list[i])/i_list[i] for i in range(len(i_list))]
    R_err = [np.sqrt(((U_err[i])/i_list[i])**2 
                        + (i_err[i]*(U_list[i])/i_list[i]**2)**2)
                for i in range(len(i_list))]

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

    P_el_arr = np.array(P_list)
    P_cond_synth_arr = np.array([f_cond_interpolated(T + 273.15) 
                                for T in T_list])
    P_rad_synth_arr = np.array([f_rad_interpolated(T + 273.15) 
                                for T in T_list])
    P_gas_synth_arr = np.array([P_el_arr[i] - P_cond_synth_arr[i] 
                                - P_rad_synth_arr[i]
                                for i in range(len(P_el_arr))])
    P_synth_arr = np.array([P_el_arr, P_cond_synth_arr, P_rad_synth_arr, 
                            P_gas_synth_arr])

    # Plot sim
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
        x_lst = T_arr - 273.15
        p_lst = 10**3 * power_arr_full[n_func]
        ax1.plot(x_lst, p_lst,
                ls = marker_list[n_func], label=label_list[n_func]
                ,color=color_list[n_func]
                 )

    # Plot P_synth from measurements
    label_list = [r"$P_{el}$", r"$P_{conduction\,synth}$", r"$P_{rad\,synth}$"
                    #, r"$P_{beam}$", r"$P_{beam gas}$"
                    #, r"$P_{bb cracker}$"
                    , r"$P_{missmatch}$"
                    #, r"$P_{gas\,synth}$"
                    #, r"$P_{laser}$"
                    ]
    color_list = ["C0", "C1", "C2", "C6"]
    for n_func,func in enumerate(func_list):
        x_lst = np.array(T_list)
        p_lst = 10**3 * P_synth_arr[n_func]
        ax1.plot(x_lst, p_lst,
                marker = ".", label=label_list[n_func]
                ,color=color_list[n_func], linestyle="None"
                 )

    ax1.set_ylabel(r"Power [mW]")
    ax1.set_xlabel(r"T [°C]")
    plt.xlim(x_lim)
    plt.ylim(y_lim)

    plt.grid(True)
    plt.legend(shadow=True,ncol =2)

    format_im = 'png' #'pdf' or png
    dpi = 300
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(out_dir + "compare_to_sim_{}".format(x_lim)
                + '.{}'.format(format_im),
                format=format_im, dpi=dpi)
    ax1.cla()

    # Plot Residuals (n_func = 3 == P_missmatch /background gas)
    P_gas_int_arr = np.array([f_gas_interpolated(T + 273.25) 
                                for T in T_list])

    P_residual_arr = P_gas_synth_arr - P_gas_int_arr

    fig = plt.figure(0, figsize=(8,6.5))
    ax1=plt.gca()
    n_func = 3
    x_lst = np.array(T_list)
    p_lst = 10**3 * P_residual_arr
    ax1.errorbar(x_lst, p_lst, yerr= np.array(P_err)*10**3, 
            # xerr=dT_err,
            marker = ".", label=label_list[n_func]
            ,color=color_list[n_func], linestyle="None"
                )

    x_lst = T_arr - 273.15
    p_lst = 0 * 10**3 * power_arr_full[n_func]
    ax1.plot(x_lst, p_lst,
            ls = "--", label=label_list[n_func]
            ,color=color_list[n_func]
                )
    plt.xlim(x_lim)
    plt.ylim(y_lim_res)

    ax1.set_ylabel(r"Power residuals [mW]")
    ax1.set_xlabel(r"T [°C]")
    plt.tight_layout()

    plt.grid(True)
    plt.legend(shadow=True,ncol =2)
    plt.savefig(out_dir + "residuals_{}".format(x_lim)
                + '.{}'.format(format_im),
                format=format_im, dpi=dpi)
    ax1.cla()

# Plot simulation  and data into same plot
# Reeval HACK subtract gas contribution in plot and do not show it
def plot_compare_to_sim_just_el(data_frame, out_dir,
                        T_arr=T_arr,
                        power_arr_full=power_arr_full,
                        x_lim=[0,400],
                        y_lim=[-0.2,3],
                        y_lim_res=[-0.2,0.2]
                        ):

    # Observables to P_synth_arr
    U_list = data_frame["U meas (mV)"].values.tolist()
    U_err = data_frame["err U"].values.tolist()
    i_list = data_frame["I meas (mA)"].values.tolist()
    i_err = data_frame["err I"].values.tolist()

    R_list = [(U_list[i])/i_list[i] for i in range(len(i_list))]
    R_err = [np.sqrt(((U_err[i])/i_list[i])**2 
                        + (i_err[i]*(U_list[i])/i_list[i]**2)**2)
                for i in range(len(i_list))]

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

    P_el_arr = np.array(P_list)
    P_cond_synth_arr = np.array([f_cond_interpolated(T + 273.15) 
                                for T in T_list])
    P_rad_synth_arr = np.array([f_rad_interpolated(T + 273.15) 
                                for T in T_list])
    P_gas_synth_arr = np.array([P_el_arr[i] - P_cond_synth_arr[i] 
                                - P_rad_synth_arr[i]
                                for i in range(len(P_el_arr))])
    # Reeval HACK subtract gas contribution in plot and do not show it
    P_el_arr = P_el_arr - P_gas_synth_arr
    P_synth_arr = np.array([P_el_arr, P_cond_synth_arr, P_rad_synth_arr, 
                            P_gas_synth_arr])

    # Plot sim
    #  P over T
    fig = plt.figure(0, figsize=(8,6.5))
    ax1=plt.gca()
    label_list = [r"$P_{el,\,sim}$", r"$P_{conduction,\,sim}$",
                  r"$P_{rad,\,sim}$"
                     #, r"$P_{beam}$", r"$P_{beam gas}$"
                     #, r"$P_{bb cracker}$"
                     , r"$P_{background\,gas,\,sim}$"
                     #, r"$P_{laser}$"
                     ]
    color_list = ["C0", "C1", "C2", "C6"]
    marker_list = ["-", "--", "--", "--"]
    # Reeval HACK :- 1 excludes back gas from plot
    # Reeval HACK 0 means just el is plotted, subtract power_arr index 3
    # (back gas from el) 
    for n_func,func in enumerate(func_list[:-1]):
        x_lst = T_arr - 273.15
        # HACK
        if n_func == 0:
            p_lst = 10**3 * (power_arr_full[n_func] - power_arr_full[3])
        else:
            p_lst = 10**3 * power_arr_full[n_func]
        ax1.plot(x_lst, p_lst,
                ls = marker_list[n_func], label=label_list[n_func]
                ,color=color_list[n_func]
                 )

    # Plot P_synth from measurements
    label_list = [r"$P_{el,\,meas}$", r"$P_{conduction\,synth}$", r"$P_{rad\,synth}$"
                    #, r"$P_{beam}$", r"$P_{beam gas}$"
                    #, r"$P_{bb cracker}$"
                    , r"$P_{missmatch}$"
                    #, r"$P_{gas\,synth}$"
                    #, r"$P_{laser}$"
                    ]
    color_list = ["C0", "C1", "C2", "C6"]
    # Reeval HACK 0 means just el is plotted, subtract power_arr index 3
    # (back gas from el)
    for n_func,func in enumerate([func_list[0]]):
        x_lst = np.array(T_list)
        p_lst = 10**3 * P_synth_arr[n_func]
        ax1.errorbar(x_lst, p_lst, P_err,
                marker = ".", label=label_list[n_func]
                ,color=color_list[n_func], linestyle="None"
                 )

    ax1.set_ylabel(r"Power [mW]")
    ax1.set_xlabel(r"T$_{avg}$ [°C]")
    plt.xlim(x_lim)
    plt.ylim(y_lim)


    secax2 = ax1.secondary_xaxis(-0.15, 
                                functions=(T_avg_to_T_max, T_max_to_T_avg))
    secax2.set_xlabel(r'T$_{max}$ [°C]')

    plt.grid(True)
    plt.legend(shadow=True,ncol =1)
    plt.tight_layout()

    format_im = 'png' #'pdf' or png
    dpi = 300
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(out_dir + "compare_to_sim_el_{}".format(x_lim)
                + '.{}'.format(format_im),
                format=format_im, dpi=dpi)
    ax1.cla()

    # Plot Residuals (n_func = 3 == P_missmatch /background gas)
    P_gas_int_arr = np.array([f_gas_interpolated(T + 273.25) 
                                for T in T_list])

    P_residual_arr = P_gas_synth_arr - P_gas_int_arr

    fig = plt.figure(0, figsize=(8,6.5))
    ax1=plt.gca()
    n_func = 3
    x_lst = np.array(T_list)
    p_lst = 10**3 * P_residual_arr
    ax1.errorbar(x_lst, p_lst, yerr= np.array(P_err)*10**3, 
            # xerr=dT_err,
            marker = ".", label=label_list[n_func]
            ,color=color_list[n_func], linestyle="None"
                )

    x_lst = T_arr - 273.15
    p_lst = 0 * 10**3 * power_arr_full[n_func]
    ax1.plot(x_lst, p_lst,
            ls = "--", label=label_list[n_func]
            ,color=color_list[n_func]
                )
    plt.xlim(x_lim)
    plt.ylim(y_lim_res)

    ax1.set_ylabel(r"Power residuals [mW]")
    ax1.set_xlabel(r"T [°C]")
    plt.tight_layout()

    plt.grid(True)
    plt.legend(shadow=True,ncol =2)
    plt.savefig(out_dir + "residuals_el_{}".format(x_lim)
                + '.{}'.format(format_im),
                format=format_im, dpi=dpi)
    ax1.cla()



# fit to residuals
def fit_mismatch(data_frame, out_dir,
                        T_arr=T_arr,
                        power_arr_full=power_arr_full,
                        ):
    os.makedirs(out_dir, exist_ok=True)
    # Observables to P_synth_arr
    U_list = data_frame["U meas (mV)"].values.tolist()
    U_err = data_frame["err U"].values.tolist()
    i_list = data_frame["I meas (mA)"].values.tolist()
    i_err = data_frame["err I"].values.tolist()

    R_list = [(U_list[i])/i_list[i] for i in range(len(i_list))]
    R_err = [np.sqrt(((U_err[i])/i_list[i])**2 
                        + (i_err[i]*(U_list[i])/i_list[i]**2)**2)
                for i in range(len(i_list))]

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

    P_el_arr = np.array(P_list)
    P_cond_synth_arr = np.array([f_cond_interpolated(T + 273.15) 
                                for T in T_list])
    P_rad_synth_arr = np.array([f_rad_interpolated(T + 273.15) 
                                for T in T_list])
    P_gas_synth_arr = np.array([P_el_arr[i] - P_cond_synth_arr[i] 
                                - P_rad_synth_arr[i]
                                for i in range(len(P_el_arr))])
    P_synth_arr = np.array([P_el_arr, P_cond_synth_arr, P_rad_synth_arr, 
                            P_gas_synth_arr])

    mismatch = P_gas_synth_arr
    # fit mismatch  with a combination of the interpolated functions simulated
    # heat flow functions
    def fit_func(T, c0,
                 c1,
                  c2
                 #,c3   
                ):
        f = (c0 
             #+ c1 * f_rad_interpolated(T + 273.15)
             + (c1 -emissivity)/emissivity * f_rad_interpolated(T + 273.15)
             + c2 * f_gas_interpolated(T + 273.15) 
            #+ c3 * f_cond_interpolated
            )
        return f
    T_arr_fit = np.delete(np.array(T_list), [1,32,34])
    mismatch_fit = mismatch
    mismatch_fit = np.delete(mismatch_fit, [1,32,34])
    # for i in [1,32, 34]:
    #     del T_list_fit[i] 
    popt, pcov = sp.optimize.curve_fit(fit_func, T_arr_fit, mismatch_fit,
                                       p0 = [0
                                            ,0.29
                                            ,2]
                                       #,sigma = P_err
                                       #, absolute_sigma = True
                                       )
    perr = np.sqrt(np.diag(pcov))
    print("popt: ", popt)
    print("perr: ", perr)
    print("emissivity_fit: ", popt[1])
    pressure = data_frame["Pressure (mbar)"].values.tolist()[0]
    print("pressure_factor: ", popt[2] * (0.02/pressure), "pm",
          perr[2] * (0.02/pressure) )
    T_arr = np.array(T_list)
    T_space = np.linspace(np.ndarray.min(T_arr), 
                          np.ndarray.max(T_arr), num = 100)
    fit_list  = [10**3 *fit_func(T, popt[0], 
                                    popt[1], 
                                    popt[2]
                                    ) 
                 for T in T_space]

    fig = plt.figure(0, figsize=(8,6.5))
    ax1=plt.gca()
    n_func = 3
    x_lst = np.array(T_list)
    p_lst = 10**3 * mismatch
    ax1.errorbar(x_lst, p_lst, yerr= np.array(P_err)*10**3, 
            # xerr=dT_err,
            marker = ".", label="Mismatch"
            , linestyle="None"
                )

    ax1.plot(T_space, fit_list,
            ls = "-", label="fit"
                )

    ax1.set_ylabel(r"Power mismatch [mW]")
    ax1.set_xlabel(r"T [°C]")
    plt.tight_layout()

    plt.grid(True)
    plt.legend(shadow=True,ncol =2)
    plt.savefig(out_dir + "fit_mismatch"
                + '.{}'.format(format_im),
                format=format_im, dpi=dpi)
    ax1.cla()

    fit_arr  = np.array([fit_func(T, popt[0], popt[1], 
                                    popt[2]
                                    ) 
                 for T in T_list])
    residual_arr = mismatch - fit_arr

    fig = plt.figure(0, figsize=(8,6.5))
    ax1=plt.gca()
    n_func = 3
    x_lst = np.array(T_list)
    p_lst = 10**3 * residual_arr
    ax1.errorbar(x_lst, p_lst, yerr= np.array(P_err)*10**3, 
            # xerr=dT_err,
            marker = ".", label="Mismatch"
            , linestyle="None"
                )

    # ax1.plot(T_space, fit_list,
    #         ls = "-", label="fit"
    #             )

    ax1.set_ylabel(r"mismatch residuals [mW]")
    ax1.set_xlabel(r"T [°C]")
    plt.tight_layout()

    plt.grid(True)
    plt.legend(shadow=True,ncol =2)
    plt.savefig(out_dir + "fit_residuals"
                + '.{}'.format(format_im),
                format=format_im, dpi=dpi)
    ax1.cla()


    

# Plotting Functions
def plot_R_vs_I_4wire_err(data_frame, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    U_list = data_frame["U meas (mV)"].values.tolist()
    U_err = data_frame["err U"].values.tolist()
    i_list = data_frame["I meas (mA)"].values.tolist()
    i_err = data_frame["err I"].values.tolist()

    R_list = [(U_list[i])/i_list[i] for i in range(len(i_list))]
    R_err = [np.sqrt(((U_err[i])/i_list[i])**2 
                        + (i_err[i]*(U_list[i])/i_list[i]**2)**2)
                for i in range(len(i_list))]

    P_list = [ 1e-3 * (i_list[i])**2* R_list[i] for i in range(len(i_list))]
    P_err = [ 1e-3 * np.sqrt((i_err[i] * 2* i_list[i] * R_list[i])**2 
                      + (R_err[i]*i_list[i]**2)**2)
             for i in range(len(i_list))]

    fig = plt.figure(0, figsize=(8,6.5))
    ax1=plt.gca()
    for n_lw, l_wire in enumerate(l_wire_list):
        ax1.errorbar(i_list, R_list, R_err, i_err,
                ".", label="{}".format(l_wire) + r"$cm$")

    ax1.set_ylabel(r"Resistance [$\Omega$]")
    ax1.set_xlabel(r"Current [mA]")
    plt.grid(True)
    plt.legend(shadow=True, title = "Wire Length")

    format_im = 'png' #'pdf' or png
    dpi = 300
    plt.savefig(out_dir +"R_vs_I_4wire_err"
                + '.{}'.format(format_im),
                format=format_im, dpi=dpi)
    ax1.cla()

    # Plot R vs P 
    fig = plt.figure(0, figsize=(8,6.5))
    ax1=plt.gca()
    for n_lw, l_wire in enumerate(l_wire_list):
        ax1.errorbar(P_list, R_list, R_err,P_err,
                ".", label="{}".format(l_wire) + r"$cm$")

    ax1.set_ylabel(r"Resistance [$\Omega$]")
    ax1.set_xlabel(r"Power [mW]")
    plt.grid(True)
    #plt.legend(shadow=True, title = "Wire Length")

    format_im = 'png' #'pdf' or png
    dpi = 300
    plt.savefig(out_dir +"R_vs_P_err"
                + '.{}'.format(format_im),
                format=format_im, dpi=dpi)
    ax1.cla()

    #Power vs Temp plot
def plot_P_vs_T(data_frame, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    U_list = data_frame["U meas (mV)"].values.tolist()
    U_err = data_frame["err U"].values.tolist()
    i_list = data_frame["I meas (mA)"].values.tolist()
    i_err = data_frame["err I"].values.tolist()

    R_list = [(U_list[i])/i_list[i] for i in range(len(i_list))]
    R_err = [np.sqrt(((U_err[i])/i_list[i])**2 
                        + (i_err[i]*(U_list[i])/i_list[i]**2)**2)
                for i in range(len(i_list))]

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
    plt.savefig(out_dir + "P_vs_T"
                + '.{}'.format(format_im),
                format=format_im, dpi=dpi)
    ax1.cla()

# def make_P_synth_array(data_frame):
#     U_list = data_frame["U meas (mV)"].values.tolist()
#     U_err = data_frame["err U"].values.tolist()
#     i_list = data_frame["I meas (mA)"].values.tolist()
#     i_err = data_frame["err I"].values.tolist()

#     R_list = [(U_list[i])/i_list[i] for i in range(len(i_list))]
#     R_err = [np.sqrt(((U_err[i])/i_list[i])**2 
#                         + (i_err[i]*(U_list[i])/i_list[i]**2)**2)
#                 for i in range(len(i_list))]

#     P_list = [10**-6 * (i_list[i])**2* R_list[i] for i in range(len(i_list))]
#     P_err = [10**-6 * np.sqrt((i_err[i] * 2* i_list[i] * R_list[i])**2 
#                       + (R_err[i]*i_list[i]**2)**2)
#              for i in range(len(i_list))]
#     alpha = 4.7*10**-3
#     dT_list = [(1/alpha)*(R_list[i]/R_list[0] - 1) for i in range(len(i_list))]
#     dT_err = [(1/alpha)*np.sqrt((R_err[i]/R_list[0])**2 
#                     + (R_err[0]*R_list[i]/(R_list[0]**2))**2)
#             for i in range(len(i_list))]
#     # dT_err = [5
#     #         for i in range(len(i_list))]
#     T_list  = [25 + dT_list[i] for i in range(len(i_list))]

#     P_el_arr = np.array(P_list)
#     P_cond_synth_arr = np.array([f_cond_interpolated(T + 273.25) 
#                                 for T in T_list])
#     P_rad_synth_arr = np.array([f_rad_interpolated(T + 273.25) 
#                                 for T in T_list])
#     P_gas_synth_arr = np.array([P_el_arr[i] - P_cond_synth_arr[i] 
#                                 - P_rad_synth_arr[i]
#                                 for i in range(len(P_el_arr))])
#     P_synth_arr = np.array([P_el_arr, P_cond_synth_arr, P_rad_synth_arr, 
#                             P_gas_synth_arr])

#     return P_synth_arr

# Synthetic P over T
def plot_P_vs_T_synth(data_frame, out_dir, 
                 f_cond_interpolated = f_cond_interpolated,
                 f_rad_interpolated = f_rad_interpolated,
                 x_lim = "auto",
                 y_lim = "auto"
                 ):
    os.makedirs(out_dir, exist_ok=True)
    U_list = data_frame["U meas (mV)"].values.tolist()
    U_err = data_frame["err U"].values.tolist()
    i_list = data_frame["I meas (mA)"].values.tolist()
    i_err = data_frame["err I"].values.tolist()

    R_list = [(U_list[i])/i_list[i] for i in range(len(i_list))]
    R_err = [np.sqrt(((U_err[i])/i_list[i])**2 
                        + (i_err[i]*(U_list[i])/i_list[i]**2)**2)
                for i in range(len(i_list))]

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

    fig = plt.figure(0, figsize=(8,6.5))
    ax1=plt.gca()
    label_list = [r"$P_{el}$", r"$P_{conduction\,synth}$", r"$P_{rad\,synth}$"
                     #, r"$P_{beam}$", r"$P_{beam gas}$"
                     #, r"$P_{bb cracker}$"
                     , r"$P_{missmatch}$"
                     #, r"$P_{gas\,synth}$"
                     #, r"$P_{laser}$"
                     ]
    color_list = ["C0", "C1", "C2", "C6"]
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
    plt.savefig(out_dir + "P_vs_T_synth"
                + '.{}'.format(format_im),
                format=format_im, dpi=dpi)
    plt.yscale('log')
    #plt.xscale('log')
    plt.savefig(out_dir + "log_P_vs_T_synth"
    + '.{}'.format(format_im),
    format=format_im, dpi=dpi)
    plt.yscale('log')
    plt.xscale('log')
    plt.savefig(out_dir + "loglog_P_vs_T_synth"
    + '.{}'.format(format_im),
    format=format_im, dpi=dpi)
    if x_lim is not "auto":
        plt.yscale('linear')
        plt.xscale('linear')
        plt.xlim(x_lim)
        if y_lim is not "auto":
            plt.ylim(y_lim)
        plt.savefig(out_dir + "P_vs_T_synth_{}".format(x_lim)
            + '.{}'.format(format_im),
            format=format_im, dpi=dpi)


    ax1.cla()

    # plot vs delta T
    fig = plt.figure(0, figsize=(8,6.5))
    ax1=plt.gca()
    label_list = [r"$P_{el}$", r"$P_{conduction\,synth}$", r"$P_{rad\,synth}$"
                     #, r"$P_{beam}$", r"$P_{beam gas}$"
                     #, r"$P_{bb cracker}$"
                     , r"$P_{missmatch}$"
                     #, r"$P_{gas\,synth}$"
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
        x_lst = np.array(dT_list)
        p_lst = 10**3 * P_synth_arr[n_func]
        # p_lst = (power_arr_full[n_func]/ power_arr_full[-1]
        #         * np.average(power_arr_full[-1]))
        ax1.plot(x_lst, p_lst,
                marker = ".", label=label_list[n_func]
                ,color=color_list[n_func], linestyle="None"
                 )

    ax1.set_ylabel(r"Power [mW]")
    ax1.set_xlabel(r"$\Delta$T [°C]")
    plt.grid(True)
    plt.legend(shadow=True)

    format_im = 'png' #'pdf' or png
    dpi = 300
    # plt.savefig(out_dir + "P_vs_T_synth"
    #             + '.{}'.format(format_im),
    #             format=format_im, dpi=dpi)
    plt.yscale('log')
    #plt.xscale('log')
    plt.savefig(out_dir + "log_P_vs_deltaT_synth"
    + '.{}'.format(format_im),
    format=format_im, dpi=dpi)
    plt.yscale('log')
    plt.xscale('log')
    plt.savefig(out_dir + "loglog_P_vs_deltaT_synth"
    + '.{}'.format(format_im),
    format=format_im, dpi=dpi)

    ax1.cla()


# # 1000mbar run
# file_dir = "C:\\Users\\Christian\\Documents\\StudiumPhD\\Measurements\\"
# file = file_dir + '2021-05-07_test_vac_1000mbar.csv'
# data_frame = pd.read_csv(file)
# print(data_frame.keys())
# # inputs
# test_run_name = "1000mbar"
# out_dir = plot_dir + test_run_name + os.sep
# plot_R_vs_I_4wire_err(data_frame, out_dir)
# plot_P_vs_T(data_frame, out_dir)
# plot_P_vs_T_synth(data_frame, out_dir)

# # 1000mbar run
# file_dir = "C:\\Users\\Christian\\Documents\\StudiumPhD\\Measurements\\"
# file = file_dir + '2021-05-07_test_vac_1000mbar.csv'
# data_frame = pd.read_csv(file)
# print(data_frame.keys())
# # inputs
# test_run_name = "1000mbar"
# out_dir = plot_dir + test_run_name + os.sep
# plot_R_vs_I_4wire_err(data_frame, out_dir)
# plot_P_vs_T(data_frame, out_dir)
# plot_P_vs_T_synth(data_frame, out_dir)

# # 0.04mbar run, Scroll pump
file_dir = "C:\\Users\\Christian\\Documents\\StudiumPhD\\Measurements\\"
file = file_dir + '2021-05-10_test_vac_0.04_mbar_fill.csv'
data_frame = pd.read_csv(file)
df_bakeout = data_frame[119:163]
data_frame_first_day = data_frame[0:92]
data_frame_second_day = data_frame[93:167]
data_frame_cut = data_frame[0:167]
#   .reset_index() is alternative to .values.tolist()
data_frame_post_bakeout = data_frame[168:320]  #   .reset_index()
data_frame_baked_400 = data_frame[168:218]
# print(data_frame.keys())
# print(data_frame_post_bakeout.keys())
# # inputs
# test_run_name = "0.04mbar"
# out_dir = plot_dir + test_run_name + os.sep
# plot_R_vs_I_4wire_err(data_frame_cut, out_dir)
# plot_P_vs_T(data_frame_cut, out_dir)
# plot_P_vs_T_synth(data_frame_cut, out_dir)
# # inputs
# test_run_name = "0.04mbar_first_day"
# out_dir = plot_dir + test_run_name + os.sep
# plot_R_vs_I_4wire_err(data_frame_first_day, out_dir)
# plot_P_vs_T(data_frame_first_day, out_dir)
# plot_P_vs_T_synth(data_frame_first_day, out_dir)
# # inputs
# test_run_name = "0.04mbar_second_day"
# out_dir = plot_dir + test_run_name + os.sep
# plot_R_vs_I_4wire_err(data_frame_second_day, out_dir)
# plot_P_vs_T(data_frame_second_day, out_dir)
# plot_P_vs_T_synth(data_frame_second_day, out_dir)
# # inputs
# test_run_name = "0.04mbar_baked"
# out_dir = plot_dir + test_run_name + os.sep
# plot_R_vs_I_4wire_err(data_frame_post_bakeout, out_dir)
# plot_P_vs_T(data_frame_post_bakeout, out_dir)
# plot_P_vs_T_synth(data_frame_post_bakeout, out_dir, x_lim=[0,400],
#                   y_lim=[-1,1.8])

# plot_compare_to_sim(data_frame_post_bakeout, out_dir,
#                     x_lim=[0,400],
#                     y_lim=[-0.2,3])
# plot_compare_to_sim(data_frame_post_bakeout, out_dir,
#                     x_lim=[0,720],
#                     y_lim=[-0.2,13.5],
#                     y_lim_res = [-4,0.2])
# plot_compare_to_sim(data_frame_post_bakeout, out_dir,
#                     x_lim=[0,150],
#                     y_lim=[-0.1,0.6],
#                     y_lim_res=[-0.02,0.02] )

test_run_name = "0.04mbar_baked_400"
out_dir = plot_dir + test_run_name + os.sep
df = data_frame_baked_400 
# plot_R_vs_I_4wire_err(df, out_dir)
# plot_P_vs_T(df, out_dir)
# plot_P_vs_T_synth(df, out_dir)

# plot_compare_to_sim(df, out_dir, y_lim_res=[-0.05,0.30])
# plot_compare_to_sim(df, out_dir,
#                     x_lim=[0,150],
#                     y_lim=[-0.1,0.6],
#                     y_lim_res=[-0.10,0.10] )
fit_mismatch(df, out_dir)

################
file_dir = "C:\\Users\\Christian\\Documents\\StudiumPhD\\Measurements\\"
file = file_dir + '2021-06-07_wire_3_post_300C_bakeout_fill.csv'
data_frame = pd.read_csv(file)
df = data_frame[6:44]
test_run_name = "wire_3_post_bakeout"
out_dir = plot_dir + test_run_name + os.sep
plot_R_vs_I_4wire_err(df, out_dir)
plot_P_vs_T_synth(df, out_dir)

plot_compare_to_sim(df, out_dir, y_lim_res=[-0.05,0.30])
plot_compare_to_sim(df, out_dir,
                    x_lim=[0,150],
                    y_lim=[-0.1,0.6],
                    y_lim_res=[-0.10,0.10] )
plot_compare_to_sim_just_el(df, out_dir, y_lim_res=[-0.05,0.30])
plot_compare_to_sim_just_el(df, out_dir,
                    x_lim=[0,150],
                    y_lim=[-0.1,0.6],
                    y_lim_res=[-0.10,0.10] )

fit_mismatch(df, out_dir)

df = data_frame[138:212]
#df_pre = data_frame[6:7]
#df = df_pre.append(df, ignore_index=True)
#print(df)
test_run_name = "wire_3_post_bakeout_1150mV"
out_dir = plot_dir + test_run_name + os.sep
plot_R_vs_I_4wire_err(df, out_dir)
plot_P_vs_T_synth(df, out_dir)

plot_compare_to_sim(df, out_dir, y_lim_res=[-0.05,0.30])
plot_compare_to_sim(df, out_dir,
                    x_lim=[0,150],
                    y_lim=[-0.1,0.6],
                    y_lim_res=[-0.10,0.10] )

plot_compare_to_sim_just_el(df, out_dir, y_lim_res=[-0.05,0.30])
plot_compare_to_sim_just_el(df, out_dir,
                    x_lim=[0,150],
                    y_lim=[-0.1,0.6],
                    y_lim_res=[-0.10,0.10] )
fit_mismatch(df, out_dir)