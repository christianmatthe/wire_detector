import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as mpatches
import os
import dill
from Wire_detector import Wire
from time import time

def set_Pt_vals(wire):
    #Input values relevant for platinum
    wire.rho_specific_resistance = 0.1058 * 10**-6
    wire.a_temperature_coefficient = 3.92 * 10**-3
    #heat conduction  change must be initialized:
    wire.k_heat_conductivity = 71.6
    wire.gen_k_heat_cond_function()
    wire.c_specific_heat = 133
    return

def set_W_vals(wire):
    #Input values relevant for Tungsten
    wire.rho_specific_resistance = 0.052 * 10**-6
    wire.a_temperature_coefficient = 4.7 * 10**-3
    #heat conduction  change must be initialized:
    wire.k_heat_conductivity = "interpolate_tungsten"
    wire.gen_k_heat_cond_function()

    wire.c_specific_heat = 133
    return
material_list = ["Pt", "W"]
# 


for material in material_list:
    start_time = time()
    ###
    # emissivity = 0.3
    #top_dir = "current_sims_3/0.02mbar_air/"
    ###
    #emissivity_list = [0.23,0.25,0.27]
    #emissivity_list = [0.05]
    emissivity_list = [0.3]
    pressure = 0
    for emissivity in emissivity_list:
        top_dir = (os.path.dirname(os.path.abspath(__file__)) + os.sep
                #+ "current_sims_3/0.0005mbar_air_em_{}_kint/".format(emissivity)
                + "current_sims_3/{}mbar_em_{}/".format(pressure,emissivity)
                )
        ###
        os.makedirs(top_dir, exist_ok=True)
        results_dir = top_dir + "results/"
        plot_dir = top_dir + "plots/"
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(plot_dir, exist_ok=True)
        d = 5
        i_current_list = [0.1,
                        0.5,0.7,1,1.2,1.5,
                        1.7, 1.9,
                        2.0,
                        2.2, 2.4, 2.6, 2.8,
                        3.0,
                        3.2, 3.4, 3.6, 3.8,
                        4.0,
                        4.2, 4.4, 4.6, 4.8,
                        5.2, 5.4, 5.6, 5.8,
                        # 6,7,8,9,10
                        6.0, 7.0, 8.0, 9.0, 10.0
                        ]
        #exp_list = [17]
        #exp_list = np.linspace(14,18,num = 5)  # later normalized to per cm**2 
        l_wire_list = [#5,
                        2
                    ] # in cm

        min_segment_length = 0.2*10**-3
        E_recombination = Wire().E_recombination
        for l_wire in l_wire_list:
            n_wire_elements = int(np.round(l_wire*10**-2 / min_segment_length))
            # simulate wire base temperature with beam off

            for i_current in i_current_list:
                time_before = time()
                wire_no_beam = Wire(
                    #k_heat_conductivity="interpolate_tungsten",

                    n_wire_elements = n_wire_elements,
                    #NOTE current scaling by wire diameter is outdated (probably)
                    #i_current = (d/5)**2 * i_current * 10**-3,
                    i_current = i_current * 10**-3,
                    d_wire = d * 10**-6,
                    
                    emissivity=emissivity,
                    pressure=pressure * 10**-3*10**5,
                    # m_molecular_gas= 29 * 1.674 * 10**-27, Air dominated
                    # m_molecular_gas= 2 * 1.674 * 10**-27, H2 dominated
                    #### Null out f_bb
                    A_cracker=0,
                    T_cracker = 293.15, # Null out this contribution
                    T_background=293.15,
                    ###
                    phi_beam=0, T_base=None
                    ###
                    ) 
                if material == "Pt":
                    set_Pt_vals(wire_no_beam)
                elif material == "W":
                    set_W_vals(wire_no_beam)
                else:
                    raise Exception("input valid wire material")

                # Run the Simulation
                mod = 4

                n_steps_no_beam = 30000 * mod
                n_steps = 10000 * mod
                record_steps = 1000
                time_step = 0.001 / mod

                #Fast settings
                # n_wire_elements = 50
                # mod = 2
                # n_steps_no_beam = 10000 * mod
                # n_steps = 10000 * mod
                # record_steps = 1000
                # time_step = 0.001 / mod
                # wire_no_beam.simulate(n_steps=n_steps_no_beam,
                #                     record_steps=record_steps,
                #                     time_step=time_step)
                wire_no_beam.simulate(n_steps=n_steps_no_beam,
                        record_steps=record_steps,
                        time_step=time_step)
                wire = wire_no_beam
                # wire.simulate(n_steps=n_steps_no_beam,
                #         record_steps=record_steps,
                #         time_step=time_step)
                run_name = "{}_d_{}_lw_{}_i_{}".format(
                            material, d, l_wire,i_current)
                os.makedirs(plot_dir + "signal/", exist_ok=True)
                os.makedirs(plot_dir + "R_over_t/", exist_ok=True)
                wire.plot_signal(plot_dir + "signal/{}".format(run_name))
                wire.plot_R_over_t(plot_dir + "R_over_t/{}".format(run_name))
                os.makedirs(plot_dir + "heat_flow/", exist_ok=True)
                wire.plot_heat_flow(plot_dir + "heat_flow/{}".format(run_name))

                # Make Logarithmic heatlfow plots
                os.makedirs(plot_dir + "log_heat_flow/", exist_ok=True)
                wire.plot_heat_flow(plot_dir + "log_heat_flow/{}".format(run_name),
                                    log_y=True)

                wire.save(results_dir + "{}".format(run_name))

                time_after = time()
                run_time = time_after - time_before
                print("finished run: " + run_name + "time required: " 
                        +  "{0:.2f} minutes".format(run_time/60))
                print("total time elapsed: {0:.2f} minutes".format(
                        (time() - start_time)/60.0))
