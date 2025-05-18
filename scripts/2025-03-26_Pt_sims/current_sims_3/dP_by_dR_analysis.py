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


            # # real world calib data paseted from scripts\current_sims_2\detection_threshold
            # dP_by_dR_list = [8.688423860792268e-06, 4.624699054619017e-06, 
            # 6.612044397681966e-06, 6.586753061539846e-06, 6.527406402260184e-06,
            #  6.314868741551134e-06, 6.401227999011866e-06, 6.46036519875163e-06, 6.5683062371582375e-06, 6.7413836237457735e-06, 6.7904828480029494e-06, 6.935143445622624e-06, 7.060662653766921e-06, 7.220844659962766e-06, 7.4321268195386845e-06, 7.63189858768887e-06, 7.892884700244502e-06, 7.997718914226566e-06, 8.382732936406588e-06, 8.814714589774296e-06, 9.312441651698394e-06, 9.796633889741152e-06, 1.0299385540300186e-05, 1.090404438465366e-05, 1.1469671058415067e-05, 1.2064348759036309e-05, 1.2640740986839793e-05, 1.3263005406779352e-05, 1.3935556238915521e-05, 1.4565990433333823e-05, 1.530525565219485e-05, 1.590165859190084e-05, 1.6463155106273557e-05, 1.729722427419253e-05, 1.803347223705302e-05, 1.86104635427077e-05, 1.9292994263048064e-05, 1.9885929035827225e-05, 2.0605109706430816e-05, 2.1318871989161993e-05, 2.1825279620004724e-05, 2.2811810867063713e-05, 2.3427926304721313e-05, 2.4053792721084035e-05, 2.4987546709592084e-05, 2.5538282979245076e-05, 2.6483770913540246e-05, 2.7452144186730136e-05, 2.8537799767943574e-05, 2.9664786842367784e-05, 3.089898286076095e-05, 3.295414740940163e-05, 3.5998441005065484e-05, 4.369813334263718e-05]

            # T_list =[25.65869511, 24.16824666, 27.87989726, 30.23928376, 33.39732976, 36.93592931, 41.25633958, 46.16093602, 51.19484176, 56.82147136, 62.38521145, 68.36370204, 74.56019868, 80.69455359, 87.1463499, 93.58478589, 
            # 100.0849924, 106.4503014, 112.8651064, 119.3360617, 131.9368309, 144.6199979, 156.8481528, 168.9693183, 180.7674462, 192.4001465, 203.6825697, 214.7497096, 225.5717271, 236.0281836, 246.2807454, 256.3092941, 266.3046577, 276.1275768, 285.6836477, 294.8884795, 304.0516334, 313.0955608, 321.9538692, 330.804584, 339.3705498, 347.8602464, 356.342082, 364.255934, 372.4307305, 380.3061255, 388.1521546, 395.8666117, 403.2736835, 410.6600658, 417.8198694, 424.850168, 431.6469147, 438.1394686, 444.187924, 443.5404379]

            if True:
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

