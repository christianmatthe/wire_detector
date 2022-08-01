import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as mpatches
import os
#import dill
import sys



def plot_heat_flow_and_temp(wire, filename="plots/heat_flow_and_temp", 
                            log_y = False):
    # change to function outside of Class                    
    self = wire
    # Calculate endstate of heat flow
    x_lst = [1000 * ((i + 0.5) * self.l_segment - (self.l_wire / 2))
            for i in range(self.n_wire_elements)]
    self.T_distribution = self.record_dict["T_distribution"][-1]
    factor = 1e3
    f_el_arr = [factor * self.f_el(j) for j in range(self.n_wire_elements)]
    f_conduction_arr = [factor * self.f_conduction(j) 
                        for j in range(self.n_wire_elements)]
    f_rad_arr = [factor * self.f_rad(j) for j in range(self.n_wire_elements)]
    f_beam_arr = [factor * self.f_beam(j) for j in range(self.n_wire_elements)]
    f_beam_gas_arr = [factor * self.f_beam_gas(j) 
                        for j in range(self.n_wire_elements)]
    f_bb_arr = [factor * self.f_bb(j) for j in range(self.n_wire_elements)]
    f_background_gas_arr = [factor * self.f_background_gas(j)
                            for j in range(self.n_wire_elements)]
    f_laser_arr = [factor * self.f_laser(j)
                            for j in range(self.n_wire_elements)]

    f_conduction_bodge_arr = [factor * self.f_conduction_bodge(j) 
                    for j in range(self.n_wire_elements)]

    # Setup Figure
    plt.figure(0, figsize=(8,6.5))
    #ax1=plt.gca()
    # Multi axis plot 
    gs = mpl.gridspec.GridSpec(2, 1, height_ratios=[1, 3]) 
    gs.update(#wspace=0.05
            hspace = 0.005
        )

    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])

    # if self.bodge == True:
    #     for i in range(0,25):
    #         f_el_arr[i] = 0
    #     for i in range(self.n_wire_elements - 25,self.n_wire_elements):
    #         f_el_arr[i] = 0
    ax1.plot(x_lst, f_el_arr, "-", label=r"$F_{el}$")
    #bodge_start
    try:
        if self.bodge == True:
            ax1.plot(x_lst, f_conduction_bodge_arr, "--"
                    , label=r"$F_{\mathrm{cond. piecewise}}$")
        else:
            ax1.plot(x_lst, f_conduction_arr, "--",
                        label=r"$-F_{conduction}$")
    except:
        #bodge_end
        ax1.plot(x_lst, f_conduction_arr, "--", label=r"$-F_{conduction}$")
    ax1.plot(x_lst, f_rad_arr, "--", label=r"$-F_{rad}$")
    ax1.plot(x_lst, f_beam_arr, "-", label=r"$F_{beam}$")
    ax1.plot(x_lst, f_beam_gas_arr, "-", label=r"$F_{beam \,gas}$")
    ax1.plot(x_lst, f_bb_arr, "-", label=r"$F_{bb\, cracker}$")
    ax1.plot(x_lst, f_background_gas_arr, "--"
                , label=r"$-F_{backgr. \, gas}$")
    ax1.plot(x_lst, f_laser_arr, "-"
                , label=r"$F_{laser}$")

    ax1.set_ylabel("Heat Flow [µW/mm]", fontsize = 16)
    ax1.set_xlabel(r"Wire positon [mm]", fontsize = 16)
    ax1.tick_params(axis='both', which='major', labelsize=12)
    plt.grid(True)
    plt.legend(shadow=True)

    #fancy legend:
    if True:
        h, l = ax1.get_legend_handles_labels()
        sources = [0,3,
                    4,5
                    #,7  # choose whether to inclde f_laser
                    ]
        sinks = [1,2
                    ,6
                    ]
        #TOP legend
        # l1 = ax1.legend([h[i] for i in sources], [l[i] for i in sources],
        #    #shadow = True,
        #    #framealpha = 0.5,
        #    loc = "lower left",
        #    bbox_to_anchor=(0, 1),
        #    fontsize = "large",
        #    title = "Heat Sources:",
        #    ncol = 2
        #    )
        # plt.gca().add_artist(l1)
        # l2 = ax1.legend([h[i] for i in sinks], [l[i] for i in sinks],
        #    #shadow = True,
        #    #framealpha = 0.5,
        #    loc = "lower right",
        #    bbox_to_anchor=(1, 1),
        #    fontsize = "large",
        #    title = "Heat Sinks:",
        #    ncol = 2
        #    )
        #Right Side Legend
        l1 = ax1.legend([h[i] for i in sources], [l[i] for i in sources],
            #shadow = True,
            #framealpha = 0.5,
            loc = "upper left",
            bbox_to_anchor=(1, 1),
            fontsize = 14,
            title = "Heat Sources:",
            title_fontsize = 14,
            ncol = 1
            )
        plt.gca().add_artist(l1)
        ax1.legend([h[i] for i in sinks], [l[i] for i in sinks],
            #shadow = True,
            #framealpha = 0.5,
            loc = "upper left",
            bbox_to_anchor=(1, 0.55),
            fontsize = 14,
            title = "Heat Sinks:",
            title_fontsize = 14,
            ncol = 1
            )
        #plt.tight_layout()

    if log_y == True:
        ax1.set_yscale("log")
        if filename == "plots/heat_flow_and_temp":
            filename= "plots/log_heat_flow_and_temp"


    # Plot Temperature Distribution
    x_lst = [1000 * ((i + 0.5) * self.l_segment - (self.l_wire / 2))
            for i in range(self.n_wire_elements)]
    T_beam_off = self.record_dict["T_distribution"][0]
    T_beam_on = self.record_dict["T_distribution"][-1]
    T_lst = [T_beam_off, T_beam_on]

    # ax0.plot(x_lst, T_lst[0] - 273.15, "-", label=r"Beam Off, " 
    #             + "R = {:.3f}".format(R_arr[0]) + r"$\Omega$")
    ax0.plot(x_lst, T_lst[1] - 273.15, "k-"
            , label=r"Beam On" 
            #+ "R = {:.3f}".format(R_arr[1]) + r"$\Omega$"
            )
    ax0.plot(x_lst, T_lst[0] - 273.15, "k-."
    , label=r"Beam Off" 
    #+ "R = {:.3f}".format(R_arr[1]) + r"$\Omega$"
    )
    ax0.set_ylabel("Wire \n Temperature [°C]")
    #ax1.set_xlabel(r"wire positon [mm]")
    ax0.grid(True)
    ax0.legend(loc = "upper left",
            bbox_to_anchor=(1, 1),
            fontsize = 14,
            title = "Beam State:",
            title_fontsize = 14,
            ncol = 1)

    format_im = 'png' #'pdf' or png
    dpi = 300
    plt.savefig(filename + '.{}'.format(format_im),
                format=format_im, dpi=dpi,
                bbox_inches='tight'
                )
    ax0.cla()
    ax1.cla()

# Just for canibalization
# def plot_signal(self, filename="plots/signal_plot"):
#     # Plot Temperature over Wire for start and end of simulation
#     # Plot Temperature over Wire
#     plt.figure(0, figsize=(8,6.5))
#     ax1 = plt.gca()

#     x_lst = [1000 * ((i + 0.5) * self.l_segment - (self.l_wire / 2))
#             for i in range(self.n_wire_elements)]
#     T_beam_off = self.record_dict["T_distribution"][0]
#     T_beam_on = self.record_dict["T_distribution"][-1]
#     T_lst = [T_beam_off, T_beam_on]

#     R_arr = np.zeros(2)
#     for i,T_dist in enumerate(T_lst):
#         self.T_distribution = T_dist
#         R_arr[i] = self.resistance_total()
    
#     U_delta = (R_arr[1] - R_arr[0]) * self.i_current
#     signal = (R_arr[1] - R_arr[0])/R_arr[0]

#     ax1.plot(x_lst, T_lst[0] - 273.15, "-", label=r"Beam Off, " 
#                 + "R = {:.3f}".format(R_arr[0]) + r"$\Omega$")
#     ax1.plot(x_lst, T_lst[1] - 273.15, "-", label=r"Beam On, " 
#                 + "R = {:.3f}".format(R_arr[1]) + r"$\Omega$")
                
#     ax1.set_ylabel("Temperature [°C]")
#     ax1.set_xlabel(r"wire positon [mm]")
#     plt.title(r"$d_{wire}$ = " + "{}".format(self.d_wire * 10**6) 
#                 + r"$\mu m$" +", I = " + "{}".format(self.i_current * 10**3)
#                 + r"$mA$" + r", $\phi_{beam}$ = 10^" + "{:.2f}".format(
#                 np.log10(self.phi_beam)))
#     plt.grid(True)
#     # get existing handles and labels
#     handles, labels = plt.gca().get_legend_handles_labels()
#     # create a patch with no color
#     empty_patch = mpatches.Patch(color='none', label='Extra label') 
#     handles.append(empty_patch)
#     labels.append("Signal: {:.2%}, ".format(signal) + r"$\Delta U$" 
#                     + " = {:.2f}".format(U_delta *10 **3) + " mV, ")
#     plt.legend(handles, labels, shadow=True)
    
    
#     format_im = 'png' #'pdf' or png
#     dpi = 300
#     plt.savefig(filename + '.{}'.format(format_im),
#                 format=format_im, dpi=dpi)
#     ax1.cla()

if __name__ == "__main__":
    #plot Options
    font = {#'family' : 'normal','weight' : 'bold',
            'size'   : 16
            #,'serif':['Helvetica']
            }
    mpl.rc('font', **font)


    # Plot background gas sims heat flow
    top_dir = os.path.dirname(os.path.abspath(__file__)) + "\\"
    #sys.path.append(top_dir + "..\\")
    from Wire_detector import Wire

    data_dir = ("C:\\Users\\Christian\\Documents\\StudiumPhD\\python\\"
                + "wire_detector\\scripts\\background_gas_sims\\results\\")
    plot_dir = top_dir + "heat_flow_and_temp/"
    os.makedirs(plot_dir, exist_ok=True)


    l_wire_list = [#5,
                    2.7
                ] # in cm
    exp_list = [14,16,17,18]
    T_cracker_list = [ 2400
                    #    ,2200,2000,1800,
                    #    1000,500,
                    #    300,0
                    ]
    for n_lw, l_wire in enumerate(l_wire_list):
        for n_phi, phi_exp in enumerate(exp_list):
            for n_T, T_cracker in enumerate(T_cracker_list):
                run_name = "lw_{}_phi_{}_Tc_{}".format(l_wire,phi_exp,
                                            T_cracker)
                wire = Wire()
                wire = wire.load(data_dir + run_name)
                # Add function that didn't exist
                # back when this program  was first run
                wire.gen_k_heat_cond_function()
                wire.m_molecular_gas = 2 * 1.674 * 10**-27
                wire.p_laser = 0
                ###

                plot_heat_flow_and_temp(
                    wire
                    , filename=plot_dir + run_name
                    , log_y = False)
                plot_heat_flow_and_temp(
                    wire
                    , filename=plot_dir + "log_" + run_name
                    , log_y = True)