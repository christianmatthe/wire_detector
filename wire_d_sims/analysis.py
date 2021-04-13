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


d_wire_list = [5,10,20]
i_current_list = [0.1, 1]
exp_list = np.linspace(14,20,num = 25)  # later normalized to per cm**2

for i_current in i_current_list:
    # Initialize Arrays
    U_arr = np.zeros((len(d_wire_list), len(exp_list)))
    T_max_arr = np.zeros((len(d_wire_list), len(exp_list)))
    T_avg_arr = np.zeros((len(d_wire_list), len(exp_list)))
    signal_arr = np.zeros((len(d_wire_list), len(exp_list)))
    for n_d, d in enumerate(d_wire_list):
        for n_p, phi_exp in enumerate(exp_list):
            run_name = "d_{}_i_{}_phi_{}".format(d, i_current, phi_exp)
            #TODO Include enumerates, output plots for every i_current
            wire = Wire()
            wire = wire.load(top_dir + "results\\" + run_name)
            l_beam = wire.l_beam

            U_beam_off = wire.U_wire(0)
            U_beam_on = wire.U_wire(-1)
            
            U_delta = U_arr[n_d, n_p] = U_beam_on - U_beam_off
            signal = signal_arr[n_d, n_p] = U_delta / U_beam_off

            T_max = T_max_arr[n_d, n_p] = np.amax(
                wire.record_dict["T_distribution"][-1])
            T_avg = T_avg_arr[n_d, n_p] = np.average(
                wire.record_dict["T_distribution"][-1])

            if True:
                # Calculate endstate of heat flow
                x_lst = [1000 * ((i + 0.5) * wire.l_segment - (wire.l_wire / 2))
                        for i in range(wire.n_wire_elements)]
                wire.T_distribution = wire.record_dict["T_distribution"][-1]
                f_el_arr = [wire.f_el(j) for j in range(wire.n_wire_elements)]
                f_conduction_arr = [wire.f_conduction(j) 
                                    for j in range(wire.n_wire_elements)]
                f_rad_arr = [wire.f_rad(j) for j in range(wire.n_wire_elements)]
                f_beam_arr = [wire.f_beam(j) for j in range(wire.n_wire_elements)]

                # Plot endstate of heat flow
                fig = plt.figure(0, figsize=(8,6.5))
                ax1=plt.gca()

                ax1.plot(x_lst, f_el_arr, "-", label=r"$F_{el}$")
                ax1.plot(x_lst, f_conduction_arr, "-", label=r"$F_{conduction}$")
                ax1.plot(x_lst, f_rad_arr, "-", label=r"$F_{rad}$")
                ax1.plot(x_lst, f_beam_arr, "-", label=r"$F_{beam}$")

                ax1.set_ylabel("Heat Flow [W/m]")
                ax1.set_xlabel(r"Wire positon [mm]")
                plt.grid(True)
                plt.legend(shadow=True)
                format_im = 'png' #'pdf' or png
                dpi = 300
                plt.savefig(plot_dir + "heat_flow_d_{}_i_{}_phi_{}".format(
                            d, i_current, phi_exp) + '.{}'.format(format_im),
                            format=format_im, dpi=dpi)
                ax1.cla()

        phi_list = 10**exp_list

    if False:
        # Plot delta U vs Phi in atoms/s
        A_beam = np.pi * ((l_beam * 10**2)/2)**2 # in cm**2

        fig = plt.figure(0, figsize=(8,6.5))
        ax1=plt.gca()

        for n_d, d in enumerate(d_wire_list):
            ax1.loglog(A_beam * phi_list, U_arr[n_d]*1000, "-", 
                       label="{}".format(d) 
                       + r"$\mu m$", basex=10)
        ax1.set_ylabel(r"$\Delta U$ [mV]")
        ax1.set_xlabel(r"$\Phi_{beam}$ [Atoms/s]")
        plt.grid(True)
        plt.legend(shadow=True)


        format_im = 'png' #'pdf' or png
        dpi = 300
        plt.savefig(plot_dir + "deltaU_compare_i_{}".format(i_current) 
                    + '.{}'.format(format_im),
                    format=format_im, dpi=dpi)
        ax1.cla()
    
    if False:
        # Plot delta U vs Phi in atoms/(s * cm**2)
        A_beam = np.pi * ((l_beam * 10**2)/2)**2


        fig = plt.figure(0, figsize=(8,6.5))
        ax1=plt.gca()

        for n_d, d in enumerate(d_wire_list):
            ax1.loglog(phi_list, U_arr[n_d]*1000, "-", 
                       label="{}".format(d) 
                       + r"$\mu m$", basex=10)
        ax1.set_ylabel(r"$\Delta U$ [mV]")
        ax1.set_xlabel(r"$\Phi_{beam}$ [Atoms/$cm^2 s$]")
        plt.grid(True)
        plt.legend(shadow=True)


        format_im = 'png' #'pdf' or png
        dpi = 300
        plt.savefig(plot_dir + "deltaU_compare_area_norm_i_{}".format(
                    i_current) + '.{}'.format(format_im),
                    format=format_im, dpi=dpi)
        ax1.cla()
    
    if False:
        # Plot delta T vs Phi in atoms/(s * cm**2)
        A_beam = np.pi * ((l_beam * 10**2)/2)**2


        fig = plt.figure(0, figsize=(8,6.5))
        ax1=plt.gca()

        for n_d, d in enumerate(d_wire_list):
            ax1.loglog(phi_list, T_avg_arr[n_d] - T_avg_arr[n_d][0] , "-", 
                       label="{}".format(d) 
                       + r"$\mu m$", basex=10)
        ax1.set_xlabel(r"$\Phi_{beam}$ [Atoms/$cm^2 s$]")
        ax1.set_ylabel(r"$\Delta$T [K]")
        plt.grid(True)
        plt.legend(shadow=True)


        format_im = 'png' #'pdf' or png
        dpi = 300
        plt.savefig(plot_dir + "deltaT_Phi_i_{}".format(
                    i_current) + '.{}'.format(format_im),
                    format=format_im, dpi=dpi)
        ax1.cla()

    if True:
        # Plot T vs Phi in atoms/(s * cm**2)
        A_beam = np.pi * ((l_beam * 10**2)/2)**2


        fig = plt.figure(0, figsize=(8,6.5))
        ax1=plt.gca()

        for n_d, d in enumerate(d_wire_list):
            ax1.loglog(phi_list, T_avg_arr[n_d] - 273.15 , "-", 
                       label="{}".format(d) 
                       + r"$\mu m$", basex=10)
        ax1.set_xlabel(r"$\Phi_{beam}$ [Atoms/$cm^2 s$]")
        ax1.set_ylabel(r"T [°C]")
        plt.grid(True)
        plt.legend(shadow=True)


        format_im = 'png' #'pdf' or png
        dpi = 300
        plt.savefig(plot_dir + "T_Phi_i_{}".format(
                    i_current) + '.{}'.format(format_im),
                    format=format_im, dpi=dpi)
        ax1.cla()

    if True:
        # Calculate endstate of heat flow
        x_lst = [1000 * ((i + 0.5) * wire.l_segment - (wire.l_wire / 2))
                for i in range(wire.n_wire_elements)]
        wire.T_distribution = wire.record_dict["T_distribution"][-1]
        f_el_arr = [wire.f_el(j) for j in range(wire.n_wire_elements)]
        f_conduction_arr = [wire.f_conduction(j) 
                            for j in range(wire.n_wire_elements)]
        f_rad_arr = [wire.f_rad(j) for j in range(wire.n_wire_elements)]
        f_beam_arr = [wire.f_beam(j) for j in range(wire.n_wire_elements)]

        # Plot endstate of heat flow
        fig = plt.figure(0, figsize=(8,6.5))
        ax1=plt.gca()

        ax1.plot(x_lst, f_el_arr, "-", label=r"$F_{el}$")
        ax1.plot(x_lst, f_conduction_arr, "-", label=r"$F_{conduction}$")
        ax1.plot(x_lst, f_rad_arr, "-", label=r"$F_{rad}$")
        ax1.plot(x_lst, f_beam_arr, "-", label=r"$F_{beam}$")

        ax1.set_ylabel("Heat Flow [W/m]")
        ax1.set_xlabel(r"Wire positon [mm]")
        plt.grid(True)
        plt.legend(shadow=True)
        format_im = 'png' #'pdf' or png
        dpi = 300
        plt.savefig(plot_dir + "T_Phi_i_{}".format(
                    i_current) + '.{}'.format(format_im),
                    format=format_im, dpi=dpi)
        ax1.cla()