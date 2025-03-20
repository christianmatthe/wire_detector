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

####plot Options
import matplotlib as mpl
font = {#'family' : 'normal','weight' : 'bold',
        'size'   : 16
        #,'serif':['Helvetica']
        }
mpl.rc('font', **font)
mpl.rcParams['figure.dpi'] = 400
##########

plot_dir = top_dir + "analysis/"
os.makedirs(plot_dir, exist_ok=True)
heat_flow_dir = top_dir + "heat_flow/"

wire = Wire()

# d_wire_list = [0.6,5,10,20]
# i_current_list = [0.1, 1]
# exp_list = np.linspace(14,20,num = 25)  # later normalized to per cm**2
d = 5
i_current = 1
# n_seg_lst = [10,20,50,100,200]
# n_seg_lst = [6,10,20,30, 40, 50,60, 80, 100, 120, 140, 160, 180, 200, 220, 240,
#              400,
#              1000
#              ]

tstep_lst = [1000,500,250,125,62,31,15]

# Initialize Arrays
U_arr = np.zeros(len(tstep_lst))
T_max_arr = np.zeros(len(tstep_lst))
T_avg_arr = np.zeros(len(tstep_lst))
signal_arr = np.zeros(len(tstep_lst))

R_wire_arr = np.zeros(len(tstep_lst))
dR_wire_arr = np.zeros(len(tstep_lst))
l_seg_arr = np.zeros(len(tstep_lst))
    

for n, time_step in enumerate(tstep_lst):
    run_name = "d_{}_i_{}_tstep_{}".format(d, i_current, 
                                            int(time_step))
    # run_name = "d_{}_i_{}_nseg_{}".format(d, i_current, n_seg)
    #TODO Include enumerates, output plots for every i_current
    wire = Wire()
    wire = wire.load(top_dir + "results\\" + run_name)
    l_beam = wire.l_beam

    # U_beam_off = wire.U_wire(0)
    # U_beam_on = wire.U_wire(-1)
    
    # U_delta = U_arr[n_d, n_p] = U_beam_on - U_beam_off
    # signal = signal_arr[n_d, n_p] = U_delta / U_beam_off

    # T_max = T_max_arr[n_d, n_p] = np.amax(
    #     wire.record_dict["T_distribution"][-1])
    # T_avg = T_avg_arr[n_d, n_p] = np.average(
    #     wire.record_dict["T_distribution"][-1])
    
    R0_wire = wire.resistance_total(wire.record_dict["T_distribution"][0])
    R_wire = wire.resistance_total(wire.record_dict["T_distribution"][-1])

    R_wire_arr[n] = R_wire
    l_seg_arr[n] = wire.l_segment
    dR_wire_arr[n] = R_wire - R0_wire

if True:
    # Plot R_wire vs n_seg

    fig = plt.figure(0, figsize=(8,6.5))
    ax1=plt.gca()

    ax1.plot(tstep_lst, R_wire_arr, "x",
             ms = 10, 
                # label="{}".format(d) 
                # + r"$\mu m$"
                )
    ax1.set_ylabel(r"$R$ [$\Omega$]")
    ax1.set_xlabel(r"$r_{\rm step}$ ")
    plt.grid(True)
    #plt.legend(shadow=True)
    plt.tight_layout()


    format_im = 'png' #'pdf' or png
    dpi = 300
    plt.savefig(plot_dir + "R_vs_tstep".format() 
                + '.{}'.format(format_im),
                format=format_im, dpi=dpi)
    ax1.cla()

if True:
    # Plot R_wire vs n_seg

    fig = plt.figure(0, figsize=(8,6.5))
    ax1=plt.gca()

    ax1.plot(tstep_lst, R_wire_arr - R_wire_arr[-1] , ".", 
             ms = 10,
                # label="{}".format(d) 
                # + r"$\mu m$"
                )
    ax1.set_ylabel(r"$R_{diff}$ [$\Omega$]")
    ax1.set_xlabel(r"$t_{\rm step}$ ")
    plt.grid(True)
    #plt.legend(shadow=True)


    format_im = 'png' #'pdf' or png
    dpi = 300
    plt.savefig(plot_dir + "R_diff_vs_tstep".format() 
                + '.{}'.format(format_im),
                format=format_im, dpi=dpi,  bbox_inches="tight")
    
    ax1.set_yscale("log")
    plt.savefig(plot_dir + "R_diff_vs_tstep_log".format() 
            + '.{}'.format(format_im),
            format=format_im, dpi=dpi,  bbox_inches="tight")
    ax1.set_xscale("log")
    plt.savefig(plot_dir + "R_diff_vs_tstep_loglog".format() 
            + '.{}'.format(format_im),
            format=format_im, dpi=dpi,  bbox_inches="tight")
    ax1.cla()

# if True:
#     # Plot R_wire vs n_seg

#     fig = plt.figure(0, figsize=(8,6.5))
#     ax1=plt.gca()

#     ax1.plot(n_seg_lst, (R_wire_arr - R_wire_arr[-1])/R_wire_arr[-1] , ".", 
#              ms = 10,
#                 # label="{}".format(d) 
#                 # + r"$\mu m$"
#                 )
#     ax1.set_ylabel(r"$R_{diff}$ [$\Omega$ / $\Omega$ ]")
#     ax1.set_xlabel(r"$n_{\rm seg}$ ")
#     plt.grid(True)
#     #plt.legend(shadow=True)
#     plt.tight_layout()


#     format_im = 'png' #'pdf' or png
#     dpi = 300
#     plt.savefig(plot_dir + "R_prop_vs_nseg".format() 
#                 + '.{}'.format(format_im),
#                 format=format_im, dpi=dpi)
    

#     ax1.set_yscale("log")
#     plt.savefig(plot_dir + "R_prop_vs_nseg_log".format() 
#                 + '.{}'.format(format_im),
#                 format=format_im, dpi=dpi)
#     ax1.cla()
#     ax1.cla()



# if True:
#     # Plot R_wire vs l_seg

#     fig = plt.figure(0, figsize=(8,6.5))
#     ax1=plt.gca()

#     ax1.plot(l_seg_arr * 10**3, (dR_wire_arr - dR_wire_arr[-1])/dR_wire_arr[-1]
#               , ".", 
#              ms = 10,
#                 # label="{}".format(d) 
#                 # + r"$\mu m$"
#                 )
#     ax1.set_ylabel(r"$\Delta R_{diff}$ [$\Omega$ / $\Omega$ ]")
#     ax1.set_xlabel(r"$l_{\rm seg} [\rm mm]$ ")
#     plt.grid(True)
#     #plt.legend(shadow=True)
#     plt.tight_layout()


#     format_im = 'png' #'pdf' or png
#     dpi = 300
#     plt.savefig(plot_dir + "dR_prop_vs_lseg".format() 
#                 + '.{}'.format(format_im),
#                 format=format_im, dpi=dpi)
    

#     ax1.set_yscale("log")
#     plt.savefig(plot_dir + "dR_prop_vs_lseg_log".format() 
#                 + '.{}'.format(format_im),
#                 format=format_im, dpi=dpi)
    
#     ax1.set_yscale("log")
#     ax1.set_xscale("log")
#     plt.savefig(plot_dir + "dR_prop_vs_lseg_loglog".format() 
#                 + '.{}'.format(format_im),
#                 format=format_im, dpi=dpi)
#     # With "fit"
#     ax1.plot(l_seg_arr * 10**3,
#               (dR_wire_arr[0] - dR_wire_arr[-1]) * (
#                   l_seg_arr**2/l_seg_arr[0]**2)/dR_wire_arr[-1], "r-"
#               )
#     ax1.set_ylabel(r"$\Delta R_{diff} / R_{diff}$ [$\Omega$ / $\Omega$ ]")
#     plt.savefig(plot_dir + "dR_prop_vs_lseg_loglog_trend".format() 
#             + '.{}'.format(format_im),
#             format=format_im, dpi=dpi)


#     ax1.cla()

# # Custom Signal plot for thesis
# if True:
# # def plot_signal(self, filename="plots/signal_plot"):
#     run_name = "d_{}_i_{}_nseg_{}".format(5, 1, 1000)
#     filename= plot_dir + "signal_custom" + "_" + run_name
#     wire = wire.load(top_dir + "results\\" + run_name)
#     # Plot Temperature over Wire for start and end of simulation
#     # Plot Temperature over Wire
#     plt.figure(0, figsize=(8,6.5))
#     ax1 = plt.gca()

#     x_lst = [1000 * ((i + 0.5) * wire.l_segment - (wire.l_wire / 2))
#             for i in range(wire.n_wire_elements)]
#     T_beam_off = wire.record_dict["T_distribution"][0]
#     T_beam_on = wire.record_dict["T_distribution"][-1]
#     T_lst = [T_beam_off, T_beam_on]

#     R_arr = np.zeros(2)
#     for i,T_dist in enumerate(T_lst):
#         wire.T_distribution = T_dist
#         R_arr[i] = wire.resistance_total()
    
#     R_delta = R_arr[1] - R_arr[0]
#     U_delta = (R_arr[1] - R_arr[0]) * wire.i_current
#     signal = (R_arr[1] - R_arr[0])/R_arr[0]

#     ax1.plot(x_lst, T_lst[0] - 273.15, "-", label=r"Initial, " 
#                 + "R = {:.3f}".format(R_arr[0]) + r"$\Omega$")
#     ax1.plot(x_lst, T_lst[1] - 273.15, "-", label=r"Final,  " 
#                 + "R = {:.3f}".format(R_arr[1]) + r"$\Omega$")
                
#     ax1.set_ylabel("Temperature [°C]")
#     ax1.set_xlabel(r"Position Along Wire [mm]")
#     # plt.title(r"$d_{wire}$ = " + "{}".format(wire.d_wire * 10**6) 
#     #             + r"$\mu m$" +", I = " + "{}".format(wire.i_current * 10**3)
#     #             + r"$mA$" + r", $\phi_{beam}$ = 10^" + "{:.2f}".format(
#     #             np.log10(wire.phi_beam)))
#     plt.grid(True)
#     # get existing handles and labels
#     handles, labels = plt.gca().get_legend_handles_labels()
#     # create a patch with no color
#     empty_patch = mpatches.Patch(color='none', label='Extra label') 
#     handles.append(empty_patch)
#     labels.append(r"$\Delta R$" 
#                     + " = {:.3f}".format(R_delta) + r"$\Omega $")
#     handles.append(empty_patch)
#     labels.append("Rel. Signal: {:.2%}".format(signal))
#     plt.legend(handles, labels, shadow=True)
    
#     plt.tight_layout()
#     format_im = 'png' #'pdf' or png
#     dpi = 300
#     plt.savefig(filename + '.{}'.format(format_im),
#                 format=format_im, dpi=dpi)
#     ax1.cla()
    
# # combined l_segment plot wit residuals
# if True:
#     # Plot R_wire vs l_seg

#     fig = plt.figure(0, figsize=(8,6.5))
#     ax1=plt.gca()

#     ax1.plot(l_seg_arr * 10**3, (dR_wire_arr - dR_wire_arr[-1])
#               , ".", 
#              ms = 10,
#              label = "sim values"
#                 # label="{}".format(d) 
#                 # + r"$\mu m$"
#                 )
#     ax1.set_ylabel(
#         r"$\Delta R - \lim_{l_{\rm seg} \to 0}(\Delta R)$ [$\Omega$ ]")
#     ax1.set_xlabel(r"$l_{\rm seg} [\rm mm]$ ")
#     plt.grid(True)
#     plt.legend(shadow=True)
#     plt.tight_layout()


#     format_im = 'png' #'pdf' or png
#     dpi = 300
#     # plt.savefig(plot_dir + "dR_prop_vs_lseg".format() 
#     #             + '.{}'.format(format_im),
#     #             format=format_im, dpi=dpi)
    

#     # ax1.set_yscale("log")
#     # plt.savefig(plot_dir + "dR_prop_vs_lseg_log".format() 
#     #             + '.{}'.format(format_im),
#     #             format=format_im, dpi=dpi)
    
#     ax1.set_yscale("log")
#     ax1.set_xscale("log")
#     plt.savefig(plot_dir + "dR_vs_lseg_loglog".format() 
#                 + '.{}'.format(format_im),
#                 format=format_im, dpi=dpi)
#     # With "fit"
#     ax1.plot(l_seg_arr * 10**3,
#               (dR_wire_arr[0] - dR_wire_arr[-1]) * (
#                   l_seg_arr**2/l_seg_arr[0]**2), "r-",
#                   label = r"$\rm{const.} \cdot {l_{\rm seg}^{ 2}}$"
#               )
#     plt.legend(shadow=True)
#     plt.savefig(plot_dir + "dR_vs_lseg_loglog_trend".format() 
#             + '.{}'.format(format_im),
#             format=format_im, dpi=dpi)


#     ax1.cla()


# # Custom R over T
# if True:
# # def plot_R_over_t(self, filename="plots/R_over_t"):
#     run_name = "d_{}_i_{}_nseg_{}".format(5, 1, 100)
#     filename= plot_dir + "R_over_T_custom" + "_" + run_name
#     wire = wire.load(top_dir + "results\\" + run_name)
#     # Plot Resistance over time
#     plt.figure(0, figsize=(8,6.5))
#     ax1=plt.gca()

#     t_lst = wire.record_dict["time"]
#     steps = len(t_lst)
#     R_lst = [wire.resistance_total(wire.record_dict["T_distribution"][i])
#                 for i in range(steps)]

#     R_tau = R_lst[0] + (R_lst[-1] - R_lst[0])*(1 - 1/np.exp(1))
#     R_tau_lst = [R_tau for i in range(len(t_lst))]
#     R_95 = R_lst[0] + (R_lst[-1] - R_lst[0])*0.95
#     R_95_lst = [R_95 for i in range(len(t_lst))]
#     # calculate time at which these are reached
#     t_tau = t_lst[np.argmin(np.absolute(R_lst - R_tau))]
#     t_95 = t_lst[np.argmin(np.absolute(R_lst - R_95))]

#     ax1.plot(t_lst, R_lst, "-", label="Resistance")
#     ax1.plot(t_lst, R_tau_lst, "-",
#                 label=r"$\Delta R \cdot$(1 - 1/e)" 
#                         + ", t = {:.3f}".format(t_tau))
#     ax1.plot(t_lst, R_95_lst, "-",
#                 label=r"$0.95 \cdot \Delta R$" + ", t = {:.3f}".format(t_95))
#     ax1.set_ylabel(r"Resistance [$\Omega$]")
#     ax1.set_xlabel(r"time [s]")
#     plt.grid(True)
#     plt.legend(shadow=True)
#     plt.tight_layout()

#     format_im = 'png' #'pdf' or png
#     dpi = 300
#     plt.savefig(filename + '.{}'.format(format_im),
#                 format=format_im, dpi=dpi)
#     ax1.cla()

#     # Residual plot
#     run_name = "d_{}_i_{}_nseg_{}".format(5, 1, 100)
#     filename= plot_dir + "R_over_T_res" + "_" + run_name
#     wire = wire.load(top_dir + "results\\" + run_name)
#     plt.figure(0, figsize=(8,6.5))
#     ax1=plt.gca()

#     t_lst = wire.record_dict["time"]
#     steps = len(t_lst)
#     R_lst = [wire.resistance_total(wire.record_dict["T_distribution"][i])
#                 for i in range(steps)]
#     R_lst_res = np.array(R_lst) - R_lst[-1]
#     R_lst = -1 * R_lst_res

#     R_tau = R_lst[0] + (R_lst[-1] - R_lst[0])*(1 - 1/np.exp(1))
#     R_tau_lst = [R_tau for i in range(len(t_lst))]
#     R_95 = R_lst[0] + (R_lst[-1] - R_lst[0])*0.95
#     R_95_lst = [R_95 for i in range(len(t_lst))]
#     # calculate time at which these are reached
#     t_tau = t_lst[np.argmin(np.absolute(R_lst - R_tau))]
#     t_95 = t_lst[np.argmin(np.absolute(R_lst - R_95))]

#     ax1.plot(t_lst, R_lst, "-", label="Resistance")
#     ax1.plot(t_lst, R_tau_lst, "-",
#                 label=r"$\Delta R \cdot$(1 - 1/e)" 
#                         + ", t = {:.3f}".format(t_tau))
#     ax1.plot(t_lst, R_95_lst, "-",
#                 label=r"$0.95 \cdot \Delta R$" + ", t = {:.3f}".format(t_95))
#     ax1.set_ylabel(r"$\Delta R$ [$\Omega$]")
#     ax1.set_xlabel(r"time [s]")
#     plt.grid(True)
#     plt.legend(shadow=True)
#     # plt.autoscale()

#     ax1.set_yscale("log")
#     format_im = 'png' #'pdf' or png
#     dpi = 300
#     plt.savefig(filename + '.{}'.format(format_im),
#                 format=format_im, dpi=dpi,  bbox_inches="tight")
#     ax1.cla()

#     #Combined plot
#     run_name = "d_{}_i_{}_nseg_{}".format(5, 1, 100)
#     filename= plot_dir + "R_over_T_combined" + "_" + run_name
#     wire = wire.load(top_dir + "results\\" + run_name)

#     fig = plt.figure(0, figsize=(8,6.5), dpi =300)
#     ax1=plt.gca()
#     gs = mpl.gridspec.GridSpec(2, 1, height_ratios=[2, 1.5]) 
#     gs.update(#wspace=0.05
#             hspace = 0.005
#         )

#     ax1 = plt.subplot(gs[0])
#     ax2 = plt.subplot(gs[1])

#     t_lst = wire.record_dict["time"]
#     steps = len(t_lst)
#     R_lst = [wire.resistance_total(wire.record_dict["T_distribution"][i])
#                 for i in range(steps)]

#     R_tau = R_lst[0] + (R_lst[-1] - R_lst[0])*(1 - 1/np.exp(1))
#     R_tau_lst = [R_tau for i in range(len(t_lst))]
#     #R_95 = R_lst[0] + (R_lst[-1] - R_lst[0])*0.95
#     R_95 = R_lst[0] + (R_lst[-1] - R_lst[0])*(1 - np.exp(-3))
#     R_95_lst = [R_95 for i in range(len(t_lst))]
#     # calculate time at which these are reached
#     t_tau = t_lst[np.argmin(np.absolute(R_lst - R_tau))]
#     t_95 = t_lst[np.argmin(np.absolute(R_lst - R_95))]

#     ax1.plot(t_lst, R_lst, "-", label="Resistance")
#     ax1.plot(t_lst, R_tau_lst, "-",
#                 label=r"$R_0 + (1 - 1/e) \cdot \Delta R$" 
#                         + r", $t_{\rm intersect}$"
#                         + " = {:.3f}".format(t_tau))
#     ax1.plot(t_lst, R_95_lst, "-",
#                 label=r"$R_0 + 0.95 \cdot \Delta R$" 
#                     + r",       $t_{\rm intersect}$"
#                     +" = {:.3f}".format(t_95))
#     ax1.set_ylabel(r"Resistance [$\Omega$]")
#     ax1.set_xlabel(r"time [s]")
#     ax1.grid(True)
#     ax1.legend(shadow=True)

#     # ax2
#     R_lst_res = np.array(R_lst[:len(R_lst)//2 + 1000]) - R_lst[-1]
#     t_lst = t_lst[:len(R_lst)//2 + 1000]
#     R_lst = -1 * R_lst_res

#     ax2.set_yscale("log")
#     ax2.plot(t_lst, R_lst, "-", label="Resistance")
#     # ax2.plot(t_lst, R_tau_lst, "-",
#     #             label=r"$\Delta R \cdot$(1 - 1/e)" 
#     #                     + ", t = {:.3f}".format(t_tau))
#     # ax2.plot(t_lst, R_95_lst, "-",
#     #             label=r"$0.95 \cdot \Delta R$" + ", t = {:.3f}".format(t_95))
#     ax2.set_ylabel(r"Remaining $\Delta R$ [$\Omega$]")
#     ax2.set_xlabel(r"time [s]")
#     ax2.grid(True)

#     #make custom pruning of uppper tick (do not plot ticks in upper 10%)
#     #so that ax2 tick does nto interfere with  ax1 tick
#     # ax2.locator_params(axis="y", min_n_ticks = 3
#     #                     )
#     y_loc = ax2.yaxis.get_majorticklocs()
#     x_loc = ax1.xaxis.get_majorticklocs()
#     #print("y_loc: ", y_loc)
#     #print("y_loc[1:-2]: ", y_loc[1:-2])
#     #print("ylim: ", ax2.get_ylim())
#     y2_min, y2_max = ax2.get_ylim()
#     y_loc = [y for y in y_loc if y2_min < y < y2_max - (y2_max -
#                                                             y2_min)*0.1]
#     #print("y_loc: ", y_loc)
#     ax2.set_yticks(y_loc)
#     # set  x lims:
#     x1_min, x1_max = ax1.get_xlim()
#     ax1.set_xlim(x1_min, x1_max/2)
#     ax2.set_xticks(x_loc)
#     ax2.set_xlim(ax1.get_xlim())

#     # Delete xticks on 1st axis
#     ax1.set_xticklabels([])


#     fig.tight_layout()
#     fig.subplots_adjust(left=0.2)

#     format_im = 'png' #'pdf' or png
#     dpi = 300
#     plt.savefig(filename + '.{}'.format(format_im),
#             format=format_im, dpi=dpi,  bbox_inches="tight")

#     ax1.cla()
#     fig.clf()
#     plt.close()