import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
import os

#plot Options
import matplotlib as mpl
font = {#'family' : 'normal','weight' : 'bold',
        'size'   : 16
        #,'serif':['Helvetica']
        }
mpl.rc('font', **font)

def make_list(data_frame, Q_ierr_const = False):
    U_list = data_frame["U meas (mV)"].values.tolist()
    U_err = data_frame["err U"].values.tolist()
    i_list = data_frame["I meas (mA)"].values.tolist()
    i_err = data_frame["err I"].values.tolist()
    #Cheat for connstant i_err
    if Q_ierr_const:
        i_err = [0.0005 for i in i_list]

    R_list = [(U_list[i])/i_list[i] for i in range(len(i_list))]
    R_err = [np.sqrt(((U_err[i])/i_list[i])**2 
                        + (i_err[i]*(U_list[i])/i_list[i]**2)**2)
                for i in range(len(i_list))]

    P_list = [10**-6 * (i_list[i])**2* R_list[i] for i in range(len(i_list))]
    P_err = [10**-6 * np.sqrt((i_err[i] * 2* i_list[i] * R_list[i])**2 
                      + (R_err[i]*i_list[i]**2)**2)
             for i in range(len(i_list))]
    return R_list, R_err, P_list, P_err


def power_limit(R_err, dP_per_dR):
    #flow limit in atoms per second (on the wire)
    power = R_err * dP_per_dR
    return power
    
def flow_limit(R_err, dP_per_dR):
    joules_per_electronvolt = 1.60218e-19
    energy_per_atom = (4.75/2) * joules_per_electronvolt
    #flow limit in atoms per second (on the wire)
    flow = R_err * dP_per_dR / (energy_per_atom) 
    return flow

def SNR(flow_sccm, cracking_efficiency, l_illuminated, flux_density_detect):
    atoms_per_sccm = 4.477962e17
    atom_flux_density = (flow_sccm * atoms_per_sccm * cracking_efficiency 
    / (np.pi * (l_illuminated/2) ** 2 ))
    return atom_flux_density/flux_density_detect 

def make_flux_limit_list(df, Q_ierr_const = False):
    d_wire = 5e-6
    l_illuminated = 1.6e-2
    A_illuminated = d_wire * l_illuminated
    R_list, R_err, P_list, P_err = make_list(df, Q_ierr_const)
    dP_per_dR_list = [(P_list[i+1] - P_list[i-1])
                      / (R_list[i+1] - R_list[i-1]) 
                              for i in range(1,len(R_list)-1)]
    #print(dP_per_dR_list)
    conv = 1e-4
    power_limit_list = [1e6*power_limit(R_err[i+1], dP_per_dR_list[i])
                       for i in range(len(dP_per_dR_list))]
    #print("power limit list: ", power_limit_list)

    flow_limit_list = [flow_limit(R_err[i+1], dP_per_dR_list[i])
                       for i in range(len(dP_per_dR_list))]
    flux_limit_list = [flow / A_illuminated for flow in flow_limit_list]
    return flux_limit_list

if __name__ == "__main__":
    file_dir = "C:\\Users\\Christian\\Documents\\StudiumPhD\\Measurements\\"
    file = file_dir + '2021-07-29_Keithley.csv'

    file_rigol = file_dir + '2021-06-07_wire_3_post_300C_bakeout_fill.csv'
    data_frame_rigol = pd.read_csv(file_rigol)

    df_rigol = data_frame_rigol[6:62]
    flux_limit_list_rigol = np.array(make_flux_limit_list(df_rigol, 
                                                    Q_ierr_const= True))

    top_dir = os.path.dirname(os.path.abspath(__file__)) + os.sep
    plot_dir = top_dir + "detection_threshold_keithley" + os.sep

    data_frame = pd.read_csv(file)

    df = data_frame[28:61]
    d_wire = 5e-6
    l_illuminated = 1.6e-2
    A_illuminated = d_wire * l_illuminated

    R_list, R_err, P_list, P_err = make_list(df)
    dP_per_dR_list = [(P_list[i+1] - P_list[i-1])
                      / (R_list[i+1] - R_list[i-1]) 
                              for i in range(1,len(R_list)-1)]
    #print(dP_per_dR_list)
    #convert to cm^2
    conv = 1e-4
    power_limit_list = [1e6*power_limit(R_err[i+1], dP_per_dR_list[i])
                       for i in range(len(dP_per_dR_list))]
    #print("power limit list: ", power_limit_list)

    flow_limit_list = np.array([flow_limit(R_err[i+1], dP_per_dR_list[i])
                       for i in range(len(dP_per_dR_list))])
    flux_limit_list = np.array([flow / A_illuminated 
                                for flow in flow_limit_list])
    #flux_limit_plot = 1e-0 * np.array(flow_limit_list)
    #print(flux_limit_list)
    
    for i in ["keithley", "both"]:
        fig = plt.figure(0, figsize=(8,6.5))
        ax1=plt.gca()
        if i == "both":
            ax1.plot(df_rigol["I meas (mA)"].values.tolist()[1:-1],
                flux_limit_list_rigol * conv,
                ".", color = "C0", label = "Rigol DP832")
        ax1.plot(df["I meas (mA)"].values.tolist()[1:-1],flux_limit_list * conv,
                ".", color = "C1", label = "Keithley 2400")

        # Switch axes sides
        ax1.yaxis.tick_right()
        ax1.yaxis.set_label_position("right")
        if conv == 1e-4:
            ax1.set_ylabel(r"$\sigma_{\Phi_{\rm at}}$ "
                + r"Flux Density Detection Limit [Atoms per $\rm cm^2 s$]")
        elif conv ==1:
            ax1.set_ylabel(r"$\sigma_{\Phi_{\rm at}}$ "
                + r"Flux Density Detection Limit [Atoms per m$^2$ s]")
        ax1.set_xlabel(r"Current [mA]")
        # #print(np.asarray(df["T (°C)"].values.tolist()[1:-1]).astype(float))
        T_arr = np.asarray(df["T (°C)"].values.tolist()).astype(float)
        # ax1.plot(T_arr,flux_limit_list,
        #         ".")

        # ax1.set_ylabel(r"Flux detection Limit [Atoms per m$^2$ s]")
        # ax1.set_xlabel(r"T [°C]")

        def flux_to_P(flux):
            joules_per_electronvolt = 1.60218e-19
            energy_per_atom = (4.75/2) * joules_per_electronvolt
            return (flux/conv) * A_illuminated * energy_per_atom * 1e6

        def P_to_flux(P):
            joules_per_electronvolt = 1.60218e-19
            energy_per_atom = (4.75/2) * joules_per_electronvolt
            return P /( A_illuminated * energy_per_atom * 1e6)

        def i_to_T(x):
            f_int = interp1d(df["I meas (mA)"].values.tolist(),
                    T_arr
                    #  ,
                    #  kind = "cubic"
                    ,fill_value="extrapolate"
                    )
            return f_int(x)

        def T_to_i(x):
            f_int = interp1d(T_arr,
                            df["I meas (mA)"].values.tolist()#,
                            #,kind = "cubic"
                            ,fill_value="extrapolate"
                            )
            out  = f_int(x.astype(float))
            return out
        
        #Axes on same side
        # secax = ax1.secondary_xaxis(-0.15, functions=(i_to_T, T_to_i))
        # secax.set_xlabel('Base Temperature [°C]')

        # secax2 = ax1.secondary_yaxis(-0.15, functions=(flux_to_P, P_to_flux))
        # secax2.set_xlabel('Power [µW]')
        #secAx on  opposite side:
        secax = ax1.secondary_xaxis(1, functions=(i_to_T, T_to_i))
        secax.set_xlabel(r'Wire Operating Temperature $T_{\rm avg}$ [°C]')

        secax2 = ax1.secondary_yaxis(0, functions=(flux_to_P, P_to_flux))
        secax2.set_ylabel(r'$\sigma_{P_{\rm meas}}$' 
                          +' Resolution in Heating Power [µW]')
        

        # #HACK Set Gridlines manually
        # xlim = ax1.get_xlim()
        # for y in secax.get_yticks():
        #     ax1.plot(xlim, (y, y),
        #              lw=0.5, color = "grey")
        # ax1.set_xlim(xlim)

        # ylim = ax1.get_ylim()
        # for x in ax1.get_xticks():
        #     ax1.plot((x, x), ylim,  lw=0.5, color = "grey")
        # secax.set_ylim(ylim)

        #HACK Set Gridlines manually
        # print(secax.get_yticks())
        # for y in secax.get_yticks():
        #HAck
        xlim = ax1.get_xlim()
        for y in np.array([0.01, 0.1, 1]):
            ax1.axhline(P_to_flux(y)*conv, color="gray", zorder=-1, 
                        linestyle="-", linewidth=0.5)

        for x in ax1.get_xticks():
            ax1.axvline(x, color="gray", zorder=-1, 
                        linestyle="-", linewidth=0.5)
            
        ax1.set_xlim(xlim)


        #plt.grid(True)
        # plt.legend(shadow=True, title = "Wire Length")

        format_im = 'png' #'pdf' or png
        dpi = 300
        plt.tight_layout()
        #plt.show()
        os.makedirs(plot_dir, exist_ok=True)
        if i == "keithley":
            plt.savefig(plot_dir + "flux_vs_current"
                        + '.{}'.format(format_im),
                        format=format_im, dpi=dpi,  bbox_inches="tight")
        if i == "both":
            ax1.legend(title = "Power Supply:")
            ax1.set_yscale('log')
            plt.savefig(plot_dir + "flux_vs_current_both"
                        + '.{}'.format(format_im),
                        format=format_im, dpi=dpi,  bbox_inches="tight")
        ax1.cla()


    #SNR Plot
    flow_sccm_list = [0.1, 0.18, 2.69, 13.4, 18.96, 20]
    cracking_efficiency_list = [0.178, 0.0545, 0.02, 0.0093, 0.0069, 0.0062]
    flux_density_detect = np.amin(flux_limit_list)
    SNR_list = [SNR(flow_sccm_list[i], cracking_efficiency_list[i],
                     l_illuminated, flux_density_detect)
                for i in range(len(flow_sccm_list))]
    flux_density_detect_rigol = np.amin(flux_limit_list_rigol)
    SNR_list_rigol = [SNR(flow_sccm_list[i], cracking_efficiency_list[i],
                    l_illuminated, flux_density_detect_rigol)
            for i in range(len(flow_sccm_list))]
    #print(SNR_list)

for i in ["keithley", "both"]:
    fig = plt.figure(0, figsize=(8,6.5))
    ax1=plt.gca()
    ax1.plot(flow_sccm_list,SNR_list,
            ".", markersize=15, label = "Keithley 2400", color = "C1")
    if i == "both":
        ax1.plot(flow_sccm_list,SNR_list_rigol,
            ".", markersize=15, label = "Rigol DP832", color = "C0")

    ax1.set_xlabel(r"Flow [sccm]")
    ax1.set_ylabel(r"Predicted SNR")

    def sccm_to_atoms(sccm):
        atoms_per_sccm = 4.477962e17
        return sccm *atoms_per_sccm
    
    def atoms_to_sccm(atoms):
        atoms_per_sccm = 4.477962e17
        return atoms / atoms_per_sccm

    secax2 = ax1.secondary_xaxis(-0.15, 
                                functions=(sccm_to_atoms, atoms_to_sccm))
    secax2.set_xlabel('[Molecules per s]')

    format_im = 'png' #'pdf' or png
    dpi = 300
    plt.grid(True)
    plt.tight_layout()
    #plt.show()
    os.makedirs(plot_dir, exist_ok=True)
    if i == "keithley":
        plt.savefig(plot_dir + "SNR"
                    + '.{}'.format(format_im),
                    format=format_im, dpi=dpi)
    if i == "both":
        ax1.set_yscale('log')
        ax1.set_xscale('log')
        ax1.legend(title = "Power Supply:")
        plt.savefig(plot_dir + "SNR_{}".format(i)
                    + '.{}'.format(format_im),
                    format=format_im, dpi=dpi)
    ax1.cla()
