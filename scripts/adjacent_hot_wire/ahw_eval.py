# Evaluating measurements taken with a wire measuring heat coming  of an 
# adjacent hot emitting wire
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
import scipy.integrate as integrate
import os

#plot Options
import matplotlib as mpl
font = {#'family' : 'normal','weight' : 'bold',
        'size'   : 16
        #,'serif':['Helvetica']
        }
mpl.rc('font', **font)

def make_lists(data_frame):
    U_list = data_frame["U meas (mV)"].values.tolist()
    U_err = data_frame["err U"].values.tolist()
    i_list = data_frame["I meas (mA)"].values.tolist()
    i_err = data_frame["err I"].values.tolist()
    #Cheat for connstant i_err
    #i_err = [0.0005 for i in i_list]

    R_list = [(U_list[i])/i_list[i] for i in range(len(i_list))]
    R_err = [np.sqrt(((U_err[i])/i_list[i])**2 
                        + (i_err[i]*(U_list[i])/i_list[i]**2)**2)
                for i in range(len(i_list))]

    P_list = [10**-6 * (i_list[i])**2* R_list[i] for i in range(len(i_list))]
    P_err = [10**-6 * np.sqrt((i_err[i] * 2* i_list[i] * R_list[i])**2 
                      + (R_err[i]*i_list[i]**2)**2)
             for i in range(len(i_list))]
    return R_list, R_err, P_list, P_err

def make_lists_cold(data_frame):
    U_list = data_frame["U meas (cold-wire) (mV)"].values.tolist()
    U_err = data_frame["err U (cold-wire)"].values.tolist()
    i_list = data_frame["I meas (cold-wire) (mA)"].values.tolist()
    i_err = data_frame["err I (cold-wire)"].values.tolist()
    #Cheat for connstant i_err
    #i_err = [0.0005 for i in i_list]

    R_list = [(U_list[i])/i_list[i] for i in range(len(i_list))]
    R_err = [np.sqrt(((U_err[i])/i_list[i])**2 
                        + (i_err[i]*(U_list[i])/i_list[i]**2)**2)
                for i in range(len(i_list))]

    P_list = [10**-6 * (i_list[i])**2* R_list[i] for i in range(len(i_list))]
    P_err = [10**-6 * np.sqrt((i_err[i] * 2* i_list[i] * R_list[i])**2 
                      + (R_err[i]*i_list[i]**2)**2)
             for i in range(len(i_list))]
    return R_list, R_err, P_list, P_err


def P_of_R_Calib(P_base, R_base, P_of_R_rel):
    return lambda R: P_base * P_of_R_rel(R / R_base)

def P_abs(R_meas, P_el_meas, P_of_R):
    #return power absorbed from outside sources
    return P_of_R(R_meas) - P_el_meas

def P_abs_err(R_meas, P_el_meas, P_of_R, R_err, P_el_err):
    #return power absorbed from outside sources
    return np.sqrt((P_of_R(R_meas) *(R_err/R_meas))**2 
                    + (P_el_err**2))

def kappa_geometric(r,d,l,l_offset):
    kappa_radial = d/(2*np.pi*r)
    kappa_point = lambda x:1 - (1/np.pi)*(np.arctan(r/x) + np.arctan(r/(l-x)))
    kappa_axial, err = integrate.quad(
        lambda x: (1/(l- 2*l_offset)) * kappa_point(x)
        ,0 + l_offset , l - l_offset
                                      )
    #print(kappa_radial,kappa_axial)
    return kappa_radial * kappa_axial

def kappa_geometric_profile(r,d,l,filename=None):
    from Wire_detector import Wire
    if filename == None:
        data_dir = ("C:\\Users\\Christian\\Documents\\StudiumPhD\\python\\"
                + "wire_detector\\scripts\\current_sims_2\\results\\")
        run_name = "lw_2.7_i_10"
        filename = data_dir + run_name
    wire = Wire()
    wire = wire.load(filename)
    # Add function that didn't exist
    # back when this program  was first run
    wire.gen_k_heat_cond_function()
    wire.m_molecular_gas = 2 * 1.674 * 10**-27
    wire.p_laser = 0

    T_max = np.amax(wire.T_distribution)
    x_list = np.linspace(0,l,num = len(wire.T_distribution))
    T_int = interp1d(x_list, wire.T_distribution, kind = "cubic"
                                ,fill_value="extrapolate")

    kappa_radial = d/(2*np.pi*r)
    kappa_point = lambda x:1 - (1/np.pi)*(np.arctan(r/x) + np.arctan(r/(l-x)))
    kappa_axial, err = integrate.quad(
        lambda x: (T_int(x)**4 / T_max**4) * kappa_point(x)
        ,0 , l)
    weight, err = integrate.quad(lambda x: (T_int(x)**4 / T_max**4)
        ,0 , l)
    print(weight)
    #print(kappa_radial,kappa_axial)
    return kappa_radial * kappa_axial / weight


def P_abs_model(P_emit, kappa_geometric, emissivity):
    return kappa_geometric * emissivity * P_emit

if __name__ == "__main__":
    top_dir = os.path.dirname(os.path.abspath(__file__)) + os.sep
    plot_dir = top_dir + "ahw" + os.sep
    os.makedirs(plot_dir, exist_ok=True)

    d = 5e-6
    l = 25e-3
    l_offset = 0 # 2e-3
    emissivity = 0.42

    # 3mm run, ~1e-4 mbar
    file_dir = "C:\\Users\\Christian\\Documents\\StudiumPhD\\Measurements\\"
    file = file_dir + '2021-07-12_adjacent_hot_wire_3mm.csv'
    data_frame = pd.read_csv(file)
    do = 2 #data row offsets csv to dataframe
    df_0 = data_frame[3-do:14-do] # low precision data
    df_1 = data_frame[39-do:51-do] # high precision data
    df_calib = data_frame[16-do:37-do] # 10 to 220mV calibration data
    d_wire = 5e-6

    r_3mm = 3e-3
    kappa_geometric_3mm = kappa_geometric(r_3mm,d,l, l_offset)
    kappa_geometric_profile_3mm = kappa_geometric_profile(r_3mm,d,l)


    R_list, R_err, P_list, P_err = make_lists_cold(df_calib)
    dP_per_dR_list = [(P_list[i+1] - P_list[i-1])
                      / (R_list[i+1] - R_list[i-1]) 
                              for i in range(1,len(R_list)-1)]
    #print(dP_per_dR_list)

    # # dp/dr over R
    # fig = plt.figure(0, figsize=(8,6.5))
    # ax1=plt.gca()
    # ax1.plot(R_list[1:-1], dP_per_dR_list,
    #         ".")

    # ax1.set_xlabel(r"Resistance [$\Omega$]")
    # ax1.set_ylabel(r"$d$P/$d$R [W/$\Omega$]")

    # format_im = 'png' #'pdf' or png
    # dpi = 300
    # plt.tight_layout()
    # #plt.show()
    # plt.savefig(plot_dir + "dP_per_dR"
    #             + '.{}'.format(format_im),
    #             format=format_im, dpi=dpi)
    # ax1.cla()
    
    # #P over R
    # fig = plt.figure(0, figsize=(8,6.5))
    # ax1=plt.gca()
    # ax1.plot(R_list, 1e6*np.array(P_list),
    #         ".")

    # ax1.set_xlabel(r"Resistance [$\Omega$]")
    # ax1.set_ylabel(r"Power [µW]")

    # format_im = 'png' #'pdf' or png
    # dpi = 300
    # plt.tight_layout()
    # #plt.show()
    # plt.savefig(plot_dir + "P_over_R_calib"
    #             + '.{}'.format(format_im),
    #             format=format_im, dpi=dpi)
    # ax1.cla()

    # #P over R dimensionless around 110mV
    # fig = plt.figure(0, figsize=(8,6.5))
    # ax1=plt.gca()
    # ax1.plot(np.array(R_list)/R_list[10], np.array(P_list)/P_list[10],
    #         ".")
    # print(R_list[10])

    # ax1.set_xlabel(r"Rel. Resistance")
    # ax1.set_ylabel(r"Rel. Power")

    # format_im = 'png' #'pdf' or png
    # dpi = 300
    # plt.tight_layout()
    # #plt.show()
    # plt.savefig(plot_dir + "rel_P_over_R_calib"
    #             + '.{}'.format(format_im),
    #             format=format_im, dpi=dpi)
    # ax1.cla()

    #Calibration of Power to Resistance scaling
    P_of_R_rel = interp1d(np.array(R_list)/R_list[10], 
                          np.array(P_list)/P_list[10],
                          kind = "cubic"
                          ,fill_value="extrapolate")

    P_of_R_3mm  = P_of_R_Calib(P_base = P_list[10] , R_base = R_list[10],
                           P_of_R_rel = P_of_R_rel)
    print(P_of_R_3mm(140.784)*1e6)


    # template eval for high precision 3mm
    base_index = 0 # index 39 - do in dataframe
    R_list_high, R_err_high, P_list_high, P_err_high = make_lists(df_1)
    (R_list_high_cold, R_err_high_cold,
     P_list_high_cold, P_err_high_cold) = make_lists_cold(df_1)
    P_of_R_high = P_of_R_Calib(P_base = P_list_high_cold[0] 
                              ,R_base = R_list_high_cold[0]
                              ,P_of_R_rel = P_of_R_rel)

    P_abs_high_list = [P_abs(R_list_high_cold[i], P_list_high_cold[i]
                            ,P_of_R_high)
                       for i in range(len(R_list_high_cold))]
    P_abs_3mm_model_list = [P_abs_model(P_list_high[i], kappa_geometric_3mm
                      ,emissivity) for i in range(len(R_list_high))]
    P_abs_3mm_profile_model_list = [P_abs_model(P_list_high[i]
                      , kappa_geometric_profile_3mm, emissivity)
                       for i in range(len(R_list_high_cold))]
    P_abs_err_high_list = [P_abs_err(R_list_high_cold[i], P_list_high_cold[i]
                                ,P_of_R_high, R_err_high_cold[i]
                                ,P_err_high_cold[i])
                for i in range(len(R_list_high_cold))]

    #P_absorbed over P_hot
    fig = plt.figure(0, figsize=(8,6.5))
    ax1=plt.gca()
    ax1.plot(1e6*np.array(P_list_high), 1e6*np.array(P_abs_high_list),
            ".")

    ax1.set_xlabel(r"P_emit hot wire [µW]")
    ax1.set_ylabel(r"P_abs cold wire [µW]")
    ax1.grid()

    format_im = 'png' #'pdf' or png
    dpi = 300
    plt.tight_layout()
    #plt.show()
    plt.savefig(plot_dir + "P_abs_over_P_emit"
                + '.{}'.format(format_im),
                format=format_im, dpi=dpi)
    ax1.cla()

    # eval for low precision 3mm
    base_index = 0 # index 39 - do in dataframe
    R_list_low, R_err_low, P_list_low, P_err_low = make_lists(df_0)
    (R_list_low_cold, R_err_low_cold,
     P_list_low_cold, P_err_low_cold) = make_lists_cold(df_0)
    P_of_R_low = P_of_R_Calib(P_base = P_list_low_cold[0] 
                              ,R_base = R_list_low_cold[0]
                              ,P_of_R_rel = P_of_R_rel)

    P_abs_low_list = [P_abs(R_list_low_cold[i], P_list_low_cold[i],P_of_R_low)
                  for i in range(len(R_list_low_cold))]
    P_abs_err_low_list = [P_abs_err(R_list_low_cold[i], P_list_low_cold[i]
                                   ,P_of_R_low, R_err_low_cold[i]
                                   ,P_err_low_cold[i])
                  for i in range(len(R_list_low_cold))]

    #P_absorbed over P_hot
    fig = plt.figure(0, figsize=(8,6.5))
    ax1=plt.gca()
    ax1.plot(1e6*np.array(P_list_low), 1e6*np.array(P_abs_low_list),
            ".")

    ax1.set_xlabel(r"P_emit hot wire [µW]")
    ax1.set_ylabel(r"P_abs cold wire [µW]")
    ax1.grid()

    format_im = 'png' #'pdf' or png
    dpi = 300
    plt.tight_layout()
    #plt.show()
    plt.savefig(plot_dir + "P_abs_over_P_emit_low_precision"
                + '.{}'.format(format_im),
                format=format_im, dpi=dpi)
    ax1.cla()


    #######
    # 6mm distance
    file_dir = "C:\\Users\\Christian\\Documents\\StudiumPhD\\Measurements\\"
    file = file_dir + '2021-07-19_adjacent_hot_wire_6mm.csv'
    data_frame_6mm = pd.read_csv(file)
    df_6mm = data_frame_6mm[3-do:14-do]

    r_6mm = 6e-3
    kappa_geometric_6mm = kappa_geometric(r_6mm,d,l, l_offset)
    kappa_geometric_profile_6mm = kappa_geometric_profile(r_6mm,d,l)


    # eval for 6mm
    base_index = 0 # index 39 - do in dataframe
    R_list_6mm, R_err_6mm, P_list_6mm, P_err_6mm = make_lists(df_6mm)
    (R_list_6mm_cold, R_err_6mm_cold,
     P_list_6mm_cold, P_err_6mm_cold) = make_lists_cold(df_6mm)
    P_of_R_6mm = P_of_R_Calib(P_base = P_list_6mm_cold[0] 
                              ,R_base = R_list_6mm_cold[0]
                              ,P_of_R_rel = P_of_R_rel)

    P_abs_6mm_list = [P_abs(R_list_6mm_cold[i], P_list_6mm_cold[i],P_of_R_6mm)
                  for i in range(len(R_list_6mm_cold))]
    P_abs_6mm_model_list = [P_abs_model(P_list_6mm[i], kappa_geometric_6mm
                      ,emissivity) for i in range(len(R_list_6mm_cold))]
    P_abs_6mm_profile_model_list = [P_abs_model(P_list_6mm[i]
                      , kappa_geometric_profile_6mm, emissivity)
                       for i in range(len(R_list_6mm_cold))]
    P_abs_err_6mm_list = [P_abs_err(R_list_6mm_cold[i], P_list_6mm_cold[i]
                                   ,P_of_R_6mm, R_err_6mm_cold[i]
                                   ,P_err_6mm_cold[i])
                  for i in range(len(R_list_6mm_cold))]
    ###########
    #P_absorbed over P_hot all data sets
    fig = plt.figure(0, figsize=(8,6.5))
    ax1=plt.gca()
    ax1.errorbar(1e6*np.array(P_list_high), 1e6*np.array(P_abs_high_list),
            yerr = 1e6*np.array(P_abs_err_high_list),
            fmt = ".", label = "3mm, High Precison")
    ax1.errorbar(1e6*np.array(P_list_low), 1e6*np.array(P_abs_low_list),
            yerr = 1e6*np.array(P_abs_err_low_list),
            fmt = ".", label = "3mm, Low Precison")
    ax1.errorbar(1e6*np.array(P_list_6mm), 1e6*np.array(P_abs_6mm_list),
            yerr = 1e6*np.array(P_abs_err_6mm_list),
            fmt = ".", label = "6mm")
    ax1.plot(1e6*np.array(P_list_6mm), 1e6*np.array(P_abs_6mm_model_list),
            "-", label = "6mm model, $\epsilon$ = {}".format(emissivity)
            , color = "C2")
    ax1.plot(1e6*np.array(P_list_high), 1e6*np.array(P_abs_3mm_model_list),
            "-", label = "3mm model, $\epsilon$ = {}".format(emissivity)
            , color = "C0")

    ax1.set_xlabel(r"P_emit hot wire [µW]")
    ax1.set_ylabel(r"P_abs cold wire [µW]")
    plt.legend(shadow=True)
    ax1.grid()
    format_im = 'png' #'pdf' or png
    dpi = 300
    plt.tight_layout()
    #plt.show()
    plt.savefig(plot_dir + "P_abs_over_P_emit_all"
                + '.{}'.format(format_im),
                format=format_im, dpi=dpi)
    ax1.cla()

#P_absorbed over P_hot all data sets, accounting for temperature profile
    fig = plt.figure(0, figsize=(8,6.5))
    ax1=plt.gca()
    ax1.errorbar(1e6*np.array(P_list_high), 1e6*np.array(P_abs_high_list),
            yerr = 1e6*np.array(P_abs_err_high_list),
            fmt = ".", label = "3mm, High Precison")
    ax1.errorbar(1e6*np.array(P_list_low), 1e6*np.array(P_abs_low_list),
            yerr = 1e6*np.array(P_abs_err_low_list),
            fmt = ".", label = "3mm, Low Precison")
    ax1.errorbar(1e6*np.array(P_list_6mm), 1e6*np.array(P_abs_6mm_list),
            yerr = 1e6*np.array(P_abs_err_6mm_list),
            fmt = ".", label = "6mm")
    ax1.plot(1e6*np.array(P_list_6mm),
             1e6*np.array(P_abs_6mm_profile_model_list),
            "-", label = "6mm model, $\epsilon$ = {}".format(emissivity)
            , color = "C2")
    ax1.plot(1e6*np.array(P_list_high), 
             1e6*np.array(P_abs_3mm_profile_model_list),
            "-", label = "3mm model, $\epsilon$ = {}".format(emissivity)
            , color = "C0")

    ax1.set_xlabel(r"P_emit hot wire [µW]")
    ax1.set_ylabel(r"P_abs cold wire [µW]")
    plt.legend(shadow=True)
    ax1.grid()
    format_im = 'png' #'pdf' or png
    dpi = 300
    plt.tight_layout()
    #plt.show()
    plt.savefig(plot_dir + "P_abs_over_P_emit_all_T_prof"
                + '.{}'.format(format_im),
                format=format_im, dpi=dpi)
    ax1.cla()
    
##########################
#High pecision  keithley, 3mm
# 6mm distance
    file_dir = "C:\\Users\\Christian\\Documents\\StudiumPhD\\Measurements\\"
    file = file_dir + '2021-08-04_ahw_3mm_keithley.csv'
    data_frame_keithley = pd.read_csv(file)
    df_keithley = data_frame_keithley[5-do:19-do]

    r_keithley = 3e-3
    kappa_geometric_keithley = kappa_geometric(r_keithley,d,l, l_offset)
    kappa_geometric_profile_keithley = kappa_geometric_profile(r_keithley,d,l)


    # eval for keithley
    base_index = 0 # index 39 - do in dataframe
    (R_list_keithley, R_err_keithley,
     P_list_keithley, P_err_keithley) = make_lists(df_keithley)
    (R_list_keithley_cold, R_err_keithley_cold,
     P_list_keithley_cold, P_err_keithley_cold) = make_lists_cold(df_keithley)
    P_of_R_keithley = P_of_R_Calib(P_base = P_list_keithley_cold[0] 
                              ,R_base = R_list_keithley_cold[0]
                              ,P_of_R_rel = P_of_R_rel)

    P_abs_keithley_list = [P_abs(R_list_keithley_cold[i], 
                           P_list_keithley_cold[i],P_of_R_keithley)
                  for i in range(len(R_list_keithley_cold))]
    P_abs_keithley_model_list = [P_abs_model(P_list_keithley[i],
                       kappa_geometric_keithley
                      ,emissivity) for i in range(len(R_list_keithley_cold))]
    P_abs_keithley_profile_model_list = [P_abs_model(P_list_keithley[i]
                      , kappa_geometric_profile_keithley, emissivity)
                       for i in range(len(R_list_keithley_cold))]
    P_abs_err_keithley_list = [P_abs_err(R_list_keithley_cold[i]
                                   ,P_list_keithley_cold[i]
                                   ,P_of_R_keithley, R_err_keithley_cold[i]
                                   ,P_err_keithley_cold[i])
                  for i in range(len(R_list_keithley_cold))]


#P_absorbed over P_hot all data sets, accounting for temperature profile
    fig = plt.figure(0, figsize=(8,6.5))
    ax1=plt.gca()
#     ax1.errorbar(1e6*np.array(P_list_high), 1e6*np.array(P_abs_high_list),
#             yerr = 1e6*np.array(P_abs_err_high_list),
#             fmt = ".", label = "3mm, High Precison")
#     ax1.errorbar(1e6*np.array(P_list_low), 1e6*np.array(P_abs_low_list),
#             yerr = 1e6*np.array(P_abs_err_low_list),
#             fmt = ".", label = "3mm, Low Precison")
#     ax1.errorbar(1e6*np.array(P_list_6mm), 1e6*np.array(P_abs_6mm_list),
#             yerr = 1e6*np.array(P_abs_err_6mm_list),
#             fmt = ".", label = "6mm")
#     ax1.plot(1e6*np.array(P_list_6mm),
#              1e6*np.array(P_abs_6mm_profile_model_list),
#             "-", label = "6mm model, $\epsilon$ = {}".format(emissivity)
#             , color = "C2")
    ax1.errorbar(1e3*np.array(P_list_keithley),
                 1e6*np.array(P_abs_keithley_list),
            yerr = 1e6*np.array(P_abs_err_keithley_list),
            fmt = ".", label = "3mm, keithley")
    ax1.plot(1e3*np.array(P_list_keithley), 
             1e6*np.array(P_abs_keithley_model_list),
            "-", label = "3mm model, $\epsilon$ = {}".format(emissivity)
            , color = "C0")

    ax1.set_xlabel(r"P_emit hot wire [mW]")
    ax1.set_ylabel(r"P_abs cold wire [µW]")
    plt.legend(shadow=True)
    ax1.grid()
    format_im = 'png' #'pdf' or png
    dpi = 300
    plt.tight_layout()
    #plt.show()
    plt.savefig(plot_dir + "P_abs_over_P_emit_keithley_T_prof"
                + '.{}'.format(format_im),
                format=format_im, dpi=dpi)
    ax1.cla()


