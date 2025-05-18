import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as mpatches
import os
import dill
import sys

from Wire_detector import Wire
from time import time

top_dir = os.path.dirname(os.path.abspath(__file__)) + os.sep
plot_dir_top = top_dir + "analysis_plots/"
os.makedirs(plot_dir_top, exist_ok=True)

# Pt simulation
# dataset_name = "0mbar_air_em_0.2"
# data_dir = top_dir + dataset_name + os.sep
# plot_dir = plot_dir_top + dataset_name + os.sep
# os.makedirs(plot_dir, exist_ok=True)
# Tungsten but put  in same folder
dataset_name = "0mbar_em_0.3"
# data_dir = top_dir + "..\\..\\current_sims_3\\" + dataset_name + os.sep
data_dir = top_dir + dataset_name + os.sep
plot_dir = plot_dir_top
os.makedirs(plot_dir, exist_ok=True)


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
l_wire_list = [
                2
            ] # in cm
material_list = ["Pt", "W"]
d_list = [5] #in µm

func_list = ["f_el", "f_rad", "f_conduction", "f_beam",
             "f_beam_gas", "f_bb", "f_background_gas", "f_laser"] 


fig = plt.figure(0, figsize=(8,6.5))
for material in material_list:
    for d in d_list:
        for l_wire in l_wire_list:
            # make power array
            # and make  resistance array
            power_arr_full = np.array([{} for i in range(len(i_current_list))])
            R_arr = np.zeros(len(i_current_list))
            T_arr = np.zeros(len(i_current_list))
            for n_i, i_current in enumerate(i_current_list):
                run_name = "{}_d_{}_lw_{}_i_{}".format(
                            material, d, l_wire,i_current)
                # run_name = "lw_{0}_i_{1}".format(l_wire_list[0],i_current)
                wire = Wire().load(data_dir + "results\\" + run_name)
                R_arr[n_i] = wire.resistance_total()
                T_arr[n_i] = np.mean(wire.T_distribution)
                power_arr_full[n_i] = {func : wire.integrate_f(getattr(wire, func))
                                        for func in func_list}

                # for n_func,func in enumerate(func_list):
                #     power = wire.integrate_f(getattr(wire, func))
                #     power_arr_full[n_func, n_i] = power
                    
            print(power_arr_full)

            # make dP_by_dR array
            # note this is in between sim points
            dP_by_dR_arr = np.array([((power_arr_full[i]["f_el"] 
                                    - power_arr_full[i-1]["f_el"] )
                                    /(R_arr[i] - R_arr[i-1]))
                                    for i in range(1,len(power_arr_full))])
            T_plot_arr = np.array([(T_arr[i] + T_arr[i-1]) / 2
                                    for i in range(1,len(power_arr_full))])
            i_plot_arr = np.array([(i_current_list[i] + i_current_list[i-1]) / 2
                                    for i in range(1,len(power_arr_full))])


            ax1=plt.gca()
            #Only "low" T
            x_lst = T_plot_arr[0:9]
            y_lst = dP_by_dR_arr[0:9]
            # p_lst = (power_arr_full[n_func]/ power_arr_full[-1]
            #         * np.average(power_arr_full[-1]))
            ax1.plot(x_lst, 1e6 * y_lst,
                    "o"
                    ,label = material
                    #,color=color_list[n_f]
                        )

            ax1.set_ylabel(r"dP/dR [µW/$\Omega$]")
            ax1.set_xlabel(r"Average Temperature [K]")
            plt.grid(True)
            plt.legend(shadow=True)



            if False:
                fig = plt.figure(0, figsize=(8,6.5))
                ax1=plt.gca()
                # label_list = [r"$P_{el}$", r"$P_{conduction}$", r"$P_{rad}$"
                #                  , r"$P_{beam}$", r"$P_{beam gas}$"
                #                  , r"$P_{bb cracker}$", r"$P_{background gas}$"
                #                  , r"$P_{laser}$"]
                # color_list = ["C0", "C1", "C2", "C3", "C4", "C5"]
                
                x_lst = T_plot_arr
                y_lst = dP_by_dR_arr
                # p_lst = (power_arr_full[n_func]/ power_arr_full[-1]
                #         * np.average(power_arr_full[-1]))
                ax1.plot(x_lst, 1e6 * y_lst,
                        "-"
                        #,label=label_list[n_func]
                        #,color=color_list[n_f]
                            )

                ax1.set_ylabel(r"dP/dR [µW/$\Omega$]")
                ax1.set_xlabel(r"Average Temperature [K]")
                plt.grid(True)
                plt.legend(shadow=True)

                format_im = 'png' #'pdf' or png
                dpi = 300
                plt.savefig(plot_dir + "dPbydR_vs_T" + "_" + run_name
                            + '.{}'.format(format_im),
                            format=format_im, dpi=dpi)
                ax1.cla()

                fig = plt.figure(0, figsize=(8,6.5))
                ax1=plt.gca()
                # label_list = [r"$P_{el}$", r"$P_{conduction}$", r"$P_{rad}$"
                #                  , r"$P_{beam}$", r"$P_{beam gas}$"
                #                  , r"$P_{bb cracker}$", r"$P_{background gas}$"
                #                  , r"$P_{laser}$"]
                # color_list = ["C0", "C1", "C2", "C3", "C4", "C5"]
                
                x_lst = T_plot_arr[0:13]
                y_lst = dP_by_dR_arr[0:13]
                # p_lst = (power_arr_full[n_func]/ power_arr_full[-1]
                #         * np.average(power_arr_full[-1]))
                ax1.plot(x_lst, 1e6 * y_lst,
                        "-"
                        #,label=label_list[n_func]
                        #,color=color_list[n_f]
                            )

                ax1.set_ylabel(r"dP/dR [µW/$\Omega$]")
                ax1.set_xlabel(r"Average Temperature [K]")
                plt.grid(True)
                plt.legend(shadow=True)

                format_im = 'png' #'pdf' or png
                dpi = 300
                plt.savefig(plot_dir + "dPbydR_vs_low_T" + "_" + run_name
                            + '.{}'.format(format_im),
                            format=format_im, dpi=dpi)
                ax1.cla()

#Outside of loops
plt.grid(True)
plt.legend(shadow=True)
format_im = 'png' #'pdf' or png
dpi = 300
plt.savefig(plot_dir + "dPbydR_vs_low_T" + "_comb"
            + '.{}'.format(format_im),
            format=format_im, dpi=dpi)
ax1.cla()