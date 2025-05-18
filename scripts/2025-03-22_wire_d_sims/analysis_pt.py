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

plot_dir = top_dir + "analysis_pt/"
os.makedirs(plot_dir, exist_ok=True)
heat_flow_dir = top_dir + "heat_flow/"
os.makedirs(heat_flow_dir, exist_ok=True)


# d_wire_list = [0.6,2,5,10,20,40]
# d_wire_list = [0.6,2,5,10,20,40,80,160]
# d_wire_list = [0.6,2,5,10,20,40,80,160]
d_wire_list = [0.3,0.6,1,2,3,4, 5,10,20,40,80,160]
i_current_list = [0]
# exp_list = np.linspace(14,20,num = 25)  # later normalized to per cm**2

exp_list = np.linspace(14,19.75,num = 24)  # later normalized to per cm**2

for i_current in i_current_list:
    # Initialize Arrays
    U_arr = np.zeros((len(d_wire_list), len(exp_list)))
    T_max_arr = np.zeros((len(d_wire_list), len(exp_list)))
    T_avg_arr = np.zeros((len(d_wire_list), len(exp_list)))
    signal_arr = np.zeros((len(d_wire_list), len(exp_list)))
    for n_d, d in enumerate(d_wire_list):
        for n_p, phi_exp in enumerate(exp_list):
            run_name = "Pt_d_{}_i_{}_phi_{}".format(d, i_current, phi_exp)
            #TODO Include enumerates, output plots for every i_current
            wire = Wire()
            wire = wire.load(top_dir + "results\\" + run_name)
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

            T_max = T_max_arr[n_d, n_p] = np.amax(
                wire.record_dict["T_distribution"][-1])
            T_avg = T_avg_arr[n_d, n_p] = np.average(
                wire.record_dict["T_distribution"][-1])

            if False:
                # Calculate endstate of heat flow
                x_lst = [1000 * ((i + 0.5) * wire.l_segment - (wire.l_wire / 2))
                        for i in range(wire.n_wire_elements)]
                wire.T_distribution = wire.record_dict["T_distribution"][-1]
                f_el_arr = wire.f_el()
                f_conduction_arr = wire.f_conduction() 
                                    
                f_rad_arr = wire.f_rad() 
                f_beam_arr = wire.f_beam()

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
                plt.savefig(heat_flow_dir + "heat_flow_d_{}_i_{}_phi_{}".format(
                            d, i_current, phi_exp) + '.{}'.format(format_im),
                            format=format_im, dpi=dpi)
                ax1.cla()

        phi_list = 10**exp_list

    # if True:
    #     # Plot delta U vs Phi in atoms/s
    #     A_beam = np.pi * ((l_beam * 10**2)/2)**2 # in cm**2

    #     fig = plt.figure(0, figsize=(8,6.5))
    #     ax1=plt.gca()

    #     for n_d, d in enumerate(d_wire_list):
    #         ax1.loglog(A_beam * phi_list, U_arr[n_d], "-", 
    #                    label="{}".format(d) 
    #                    + r"$\mu m$", base=10)
    #     ax1.set_ylabel(r"$\Delta R$ [$\Omega$]")
    #     ax1.set_xlabel(r"$\Phi_{beam}$ [Atoms/s]")
    #     plt.grid(True)
    #     plt.legend(shadow=True)


    #     format_im = 'png' #'pdf' or png
    #     dpi = 300
    #     plt.savefig(plot_dir + "deltaR_compare_i_{}".format(i_current) 
    #                 + '.{}'.format(format_im),
    #                 format=format_im, dpi=dpi)
    #     ax1.cla()

    # if True:
    #     # Plot rel_signal vs Phi in atoms/s
    #     A_beam = np.pi * ((l_beam * 10**2)/2)**2 # in cm**2

    #     fig = plt.figure(0, figsize=(8,6.5))
    #     ax1=plt.gca()

    #     for n_d, d in enumerate(d_wire_list):
    #         ax1.loglog(A_beam * phi_list, signal_arr[n_d], "-", 
    #                    label="{}".format(d) 
    #                    + r"$\mu m$", base=10)
    #     ax1.set_ylabel(r"Relative Signal $\Delta R / R_{\rm initial}$")
    #     ax1.set_xlabel(r"$\Phi_{beam}$ [Atoms/s]")
    #     plt.grid(True)
    #     plt.legend(shadow=True)


    #     format_im = 'png' #'pdf' or png
    #     dpi = 300
    #     plt.savefig(plot_dir + "rel_sig_compare_i_{}".format(i_current) 
    #                 + '.{}'.format(format_im),
    #                 format=format_im, dpi=dpi)
    #     ax1.cla()

    # if True:
    #     # Plot rel_signal vs Phi in atoms/cm**2 s
    #     A_beam = np.pi * ((l_beam * 10**2)/2)**2 # in cm**2

    #     fig = plt.figure(0, figsize=(8,6.5))
    #     ax1=plt.gca()

    #     for n_d, d in enumerate(d_wire_list):
    #         ax1.loglog( phi_list, signal_arr[n_d], "-", 
    #                    label="{}".format(d) 
    #                    + r"$\mu m$", base=10)
    #     ax1.set_ylabel(r"Relative Signal $\Delta R / R_{\rm initial}$")
    #     ax1.set_xlabel(r"$\Phi_{beam}$ [Atoms/$cm^2 s$]")
    #     plt.grid(True)
    #     plt.legend(shadow=True)


    #     format_im = 'png' #'pdf' or png
    #     dpi = 300
    #     plt.savefig(plot_dir + "rel_sig_compare_area_norm_i_{}".format(
    #                 i_current) 
    #                 + '.{}'.format(format_im),
    #                 format=format_im, dpi=dpi)
    #     ax1.cla()
    
#     if True:
#         # Plot delta U vs Phi in atoms/(s * cm**2)
#         A_beam = np.pi * ((l_beam * 10**2)/2)**2


#         fig = plt.figure(0, figsize=(8,6.5))
#         ax1=plt.gca()

#         for n_d, d in enumerate(d_wire_list):
#             ax1.loglog(phi_list, U_arr[n_d]*1000, "-", 
#                        label="{}".format(d) 
#                        + r"$\mu m$", base=10)
#         ax1.set_ylabel(r"$\Delta U$ [mV]")
#         ax1.set_xlabel(r"$\Phi_{beam}$ [Atoms/$cm^2 s$]")
#         plt.grid(True)
#         plt.legend(shadow=True)


#         format_im = 'png' #'pdf' or png
#         dpi = 300
#         plt.savefig(plot_dir + "deltaU_compare_area_norm_i_{}".format(
#                     i_current) + '.{}'.format(format_im),
#                     format=format_im, dpi=dpi)
#         ax1.cla()
    
#     if True:
#         # Plot delta T vs Phi in atoms/(s * cm**2)
#         A_beam = np.pi * ((l_beam * 10**2)/2)**2


#         fig = plt.figure(0, figsize=(8,6.5))
#         ax1=plt.gca()

#         for n_d, d in enumerate(d_wire_list):
#             ax1.loglog(phi_list, T_avg_arr[n_d] - T_avg_arr[n_d][0] , "-", 
#                        label="{}".format(d) 
#                        + r"$\mu m$", base=10)
#         ax1.set_xlabel(r"$\Phi_{beam}$ [Atoms/$cm^2 s$]")
#         ax1.set_ylabel(r"$\Delta$T [K]")
#         plt.grid(True)
#         plt.legend(shadow=True)


#         format_im = 'png' #'pdf' or png
#         dpi = 300
#         plt.savefig(plot_dir + "deltaT_Phi_i_{}".format(
#                     i_current) + '.{}'.format(format_im),
#                     format=format_im, dpi=dpi)
#         ax1.cla()

#     if True:
#         # Plot T vs Phi in atoms/(s * cm**2)
#         A_beam = np.pi * ((l_beam * 10**2)/2)**2


#         fig = plt.figure(0, figsize=(8,6.5))
#         ax1=plt.gca()

#         for n_d, d in enumerate(d_wire_list):
#             ax1.loglog(phi_list, T_avg_arr[n_d] - 273.15 , "-", 
#                        label="{}".format(d) 
#                        + r"$\mu m$", base=10)
#         ax1.set_xlabel(r"$\Phi_{beam}$ [Atoms/$cm^2 s$]")
#         ax1.set_ylabel(r"T [°C]")
#         plt.grid(True)
#         plt.legend(shadow=True)


#         format_im = 'png' #'pdf' or png
#         dpi = 300
#         plt.savefig(plot_dir + "T_Phi_i_{}".format(
#                     i_current) + '.{}'.format(format_im),
#                     format=format_im, dpi=dpi)
#         ax1.cla()

# #COMbine temperature and Relative Signal Plot
#     if True:
#         import matplotlib as mpl
#         fig = plt.figure(0, figsize=(8,6.5), dpi =300)
#         ax1=plt.gca()
#         gs = mpl.gridspec.GridSpec(2, 1, height_ratios=[2, 1.5]) 
#         gs.update(#wspace=0.05
#                 hspace = 0.005
#             )

#         ax1 = plt.subplot(gs[0])
#         ax2 = plt.subplot(gs[1])

#         for n_d, d in enumerate(d_wire_list):
#             ax1.loglog( phi_list, signal_arr[n_d], "-", 
#                        label="{}".format(d) 
#                        + r"$\,\rm \mu m$", base=10)
#         ax1.set_ylabel(r"Relative Signal $\Delta R / R_{\rm initial}$")
#         ax1.set_xlabel(r"$\Phi_{beam}$ [Atoms/$\rm (cm^2 s)$]")
#         ax1.grid(True)
#         ax1.legend(shadow=True, title = "Wire Diameter:", fontsize =  14)

#         for n_d, d in enumerate(d_wire_list):
#             ax2.loglog(phi_list, T_avg_arr[n_d] - 273.15 , "-", 
#                        label="{}".format(d) 
#                        + r"$\,\rm \mu m$", base=10)
#         ax2.set_xlabel(r"$\Phi_{beam}$ [Atoms/$\rm (cm^2 s)$]")
#         ax2.set_ylabel(r"T [°C]")
#         ax2.grid(True)
#         #ax2.legend(shadow=True, title = "Wire Diameter:", fontsize =  14)

#         #make custom pruning of uppper tick (do not plot ticks in upper 10%)
#         #so that ax2 tick does nto interfere with  ax1 tick
#         # ax2.locator_params(axis="y", min_n_ticks = 3
#         #                     )
#         y_loc = ax2.yaxis.get_majorticklocs()
#         x_loc = ax1.xaxis.get_majorticklocs()
#         #print("y_loc: ", y_loc)
#         #print("y_loc[1:-2]: ", y_loc[1:-2])
#         #print("ylim: ", ax2.get_ylim())
#         y2_min, y2_max = ax2.get_ylim()
#         y_loc = [y for y in y_loc if y2_min < y < y2_max - (y2_max -
#                                                                 y2_min)*0.1]
#         #print("y_loc: ", y_loc)
#         ax2.set_yticks(y_loc)
#         # set  x lims:
#         x1_min, x1_max = ax1.get_xlim()
#         ax1.set_xlim(x1_min, x1_max/2)
#         ax2.set_xticks(x_loc)
#         ax2.set_xlim(ax1.get_xlim())

#         # Delete xticks on 1st axis
#         ax1.set_xticklabels([])


#         fig.tight_layout()
#         fig.subplots_adjust(left=0.2)

#         format_im = 'png' #'pdf' or png
#         dpi = 300
#         plt.savefig(plot_dir + "rel_sig_and T_compare_area_norm_i_{}".format(
#                     i_current)  + '.{}'.format(format_im),
#                 format=format_im, dpi=dpi,  bbox_inches="tight")

#         ax1.cla()
#         fig.clf()
#         plt.close()


#     if True:
#         # Delta R over d
#         # Plot delta U vs Phi in atoms/(s * cm**2)
#         A_beam = np.pi * ((l_beam * 10**2)/2)**2 # in cm**2

#         # Make appropriate R_arr list
#         #12th phi index is 16
#         i_phi = 12
#         dR_list = [R_arr[n_d,i_phi] for n_d, d in enumerate(d_wire_list)]

#         fig = plt.figure(0, figsize=(8,6.5))
#         ax1=plt.gca()

#         ax1.loglog(d_wire_list, dR_list, "o", 
#                     label="phi {}".format(exp_list[i_phi]) 
#                     , base=10)
#         ax1.set_ylabel(r"$\Delta R$ [$\Omega$]")
#         ax1.set_xlabel(r"Wire Diameter [$\rm \mu m$]")
#         plt.grid(True)
#         plt.legend(shadow=True)


#         format_im = 'png' #'pdf' or png
#         dpi = 300
#         plt.savefig(plot_dir + "deltaR_vs_d_phi_{}".format(exp_list[i_phi]) 
#                     + '.{}'.format(format_im),
#                     format=format_im, dpi=dpi)
#         ax1.cla()

    if True:
        # rel_signal over d

        # Make appropriate R_arr list
        #12th phi index is 16
        signal_arr_Pt = signal_arr
        d_wire_list_Pt = np.array(d_wire_list)
        for i_phi in [0,4,8,12,16,20]:
            #i_phi = 12
            sig_list = [signal_arr[n_d,i_phi] 
                        for n_d, d in enumerate(d_wire_list)]
            sig_list = np.array(sig_list)
            d_wire_list = np.array(d_wire_list)


            fig = plt.figure(0, figsize=(8,6.5))
            ax1=plt.gca()

            ax1.loglog(d_wire_list, sig_list, "o", 
                        label="phi {}".format(exp_list[i_phi]) 
                        , base=10)
            ax1.loglog(d_wire_list, sig_list[-1] * (
                        d_wire_list[-1]/d_wire_list),
                        "-", 
                        label="1/x".format(exp_list[i_phi]) 
                        , base=10)
            ax1.set_ylabel(r"Relative Signal $\Delta R / R_{\rm initial}$")
            ax1.set_xlabel(r"Wire Diameter [$\rm \mu m$]")
            plt.grid(True)
            plt.legend(shadow=True)


            format_im = 'png' #'pdf' or png
            dpi = 300
            plt.savefig(plot_dir + "rel_sig_vs_d_phi_{}".format(exp_list[i_phi]) 
                        + '.{}'.format(format_im),
                        format=format_im, dpi=dpi)
            ax1.cla()

# Now load data from W wire for comparison

# for i_current in i_current_list: Loop already happens above
    # Initialize Arrays
    U_arr = np.zeros((len(d_wire_list), len(exp_list)))
    T_max_arr = np.zeros((len(d_wire_list), len(exp_list)))
    T_avg_arr = np.zeros((len(d_wire_list), len(exp_list)))
    signal_arr = np.zeros((len(d_wire_list), len(exp_list)))
    for n_d, d in enumerate(d_wire_list):
        for n_p, phi_exp in enumerate(exp_list):
            try:
                run_name = "d_{}_i_{}_phi_{}".format(d, i_current, phi_exp)
                wire = Wire()
                wire = wire.load(top_dir + "results\\" + run_name)
            except:
                #In case d is for some reason treated as non-int for  integer 
                # values
                run_name = "d_{}_i_{}_phi_{}".format(int(d), i_current, phi_exp)
                wire = Wire()
                wire = wire.load(top_dir + "results\\" + run_name)
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

            T_max = T_max_arr[n_d, n_p] = np.amax(
                wire.record_dict["T_distribution"][-1])
            T_avg = T_avg_arr[n_d, n_p] = np.average(
                wire.record_dict["T_distribution"][-1])

    if True:
        # rel_signal over d

        # Make appropriate R_arr list
        #12th phi index is 16
        for i_phi in [0,4,8,12,16,20]:
            #i_phi = 12
            sig_list = [signal_arr[n_d,i_phi] 
                        for n_d, d in enumerate(d_wire_list)]
            sig_list = np.array(sig_list)
            d_wire_list = np.array(d_wire_list)

            sig_list_Pt = [signal_arr_Pt[n_d,i_phi] 
                for n_d, d in enumerate(d_wire_list)]
            sig_list_Pt = np.array(sig_list_Pt)


            fig = plt.figure(0, figsize=(8,6.5))
            ax1=plt.gca()

            ax1.loglog(d_wire_list, sig_list, "o", 
                        label="W".format(exp_list[i_phi]) 
                        , base=10)
            ax1.loglog(d_wire_list_Pt, sig_list_Pt, "o", 
                        label="Pt".format(exp_list[i_phi]) 
                        , base=10)
            # ax1.loglog(d_wire_list, sig_list[-1] * (
            #             d_wire_list[-1]/d_wire_list),
            #             "-", 
            #             label="1/x".format(exp_list[i_phi]) 
            #             , base=10)
            ax1.set_ylabel(r"Relative Signal $\Delta R / R_{\rm initial}$")
            ax1.set_xlabel(r"Wire Diameter [$\rm \mu m$]")
            plt.grid(True)
            plt.legend(shadow=True)


            format_im = 'png' #'pdf' or png
            dpi = 300
            plt.savefig(plot_dir + "PtW_rel_sig_vs_d_phi_{}".format(exp_list[i_phi]) 
                        + '.{}'.format(format_im),
                        format=format_im, dpi=dpi)
            ax1.cla()

            fig = plt.figure(0, figsize=(8,6.5))
            ax1=plt.gca()

            ax1.loglog(d_wire_list, sig_list, "-", 
                        label="W".format(exp_list[i_phi]) 
                        , base=10)
            ax1.loglog(d_wire_list_Pt, sig_list_Pt, "-", 
                        label="Pt".format(exp_list[i_phi]) 
                        , base=10)
            # ax1.loglog(d_wire_list, sig_list[-1] * (
            #             d_wire_list[-1]/d_wire_list),
            #             "-", 
            #             label="1/x".format(exp_list[i_phi]) 
            #             , base=10)
            ax1.set_ylabel(r"Relative Signal $\Delta R / R_{\rm initial}$")
            ax1.set_xlabel(r"Wire Diameter [$\rm \mu m$]")
            plt.grid(True)
            plt.legend(shadow=True)


            format_im = 'png' #'pdf' or png
            dpi = 300
            plt.savefig(plot_dir + "PtW_rel_sig_vs_d_phi_{}_line".format(exp_list[i_phi]) 
                        + '.{}'.format(format_im),
                        format=format_im, dpi=dpi)
            ax1.cla()