import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as mpatches
import os
import dill
from Wire_detector import Wire
from time import time



##### Constants
E_recombination = Wire().E_recombination
segment_length = Wire(n_wire_elements = 100,l_wire=2*10**-2).l_segment

emissivity = 0.08
pressure = 0
d = 5
i_current = 1
l_wire = 2
####

### scan lists
# Have to correct for 1e2 factor due to units in  which l_segment is stored
# in the Wire object
x_pos_list = np.linspace(-1,0, 40, endpoint= True) + segment_length*1e2/2
print(x_pos_list)
power_watts_list = [100e-6, 1e-6]
###

start_time = time()
###

# min_segment_length = 0.1*10**-3
# for l_wire in l_wire_list:
#     n_max = l_wire*10**-2 / min_segment_length
#     if n_max > 100:
#         n_wire_elements = 100
#     else:
#         if round(n_max,-1) == 0:
#             n_wire_elements = 10
#         else:
#             n_wire_elements = int(round(n_max,-1))
#     # simulate wire base temperature with beam off

for j, power_watts in enumerate(power_watts_list):
    for i, x_offset_beam in enumerate(x_pos_list):
        top_dir = (os.path.dirname(os.path.abspath(__file__)) + os.sep
                + "sensitivity_profile_sims/set_3_slow/{}muW_x_off_{}/".format(
                    power_watts * 1e6, i)
                )
        ###
        os.makedirs(top_dir, exist_ok=True)
        results_dir = top_dir + "results/"
        plot_dir = top_dir + "plots/"
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(plot_dir, exist_ok=True)


        #  Run the Simulation
        mod = 4
        n_wire_elements = 100
        n_steps_no_beam = 30000 * mod
        n_steps = 10000 * mod
        record_steps = 1000
        time_step = 0.001 / mod

        # #### Fast settings
        # n_wire_elements = 50
        # mod = 2
        # n_steps_no_beam = 10000 * mod
        # n_steps = 10000 * mod
        # record_steps = 1000
        # time_step = 0.001 / mod


        time_before = time()
        wire_no_beam = Wire(
            #k_heat_conductivity="interpolate_tungsten",

            n_wire_elements = n_wire_elements,
            i_current = i_current * 10**-3, d_wire = d * 10**-6,
            emissivity = emissivity, l_wire=l_wire*10**-2, 
            rho_specific_resistance=0.054 * 10**-6,
            pressure=pressure * 10**-3*10**5, # mbar to kpa
            m_molecular_gas= 1 * 1.674 * 10**-27,
            T_cracker = 298.25, # Null out this contribution
            T_background=298.25,
            ### BEAM
            # use E_rec/2 factor to get input in Watts
            phi_beam= 0 / (E_recombination/2),
            beam_shape = "Point",
            ###
            ) 
        
                        
        wire_no_beam.simulate(n_steps=n_steps_no_beam,
                            record_steps=record_steps,
                            time_step=time_step)

        wire = Wire(
            #k_heat_conductivity="interpolate_tungsten",

            n_wire_elements = n_wire_elements,
            i_current = i_current * 10**-3, d_wire = d * 10**-6,
            emissivity = emissivity, l_wire=l_wire*10**-2, 
            rho_specific_resistance=0.054 * 10**-6,
            pressure=pressure * 10**-3*10**5, # mbar to kpa
            m_molecular_gas= 1 * 1.674 * 10**-27,
            T_cracker = 298.25, # Null out this contribution
            T_background=298.25,
            ### BEAM
            # use E_rec/2 factor to get input in Watts
            phi_beam= power_watts / (E_recombination/2),
            beam_shape = "Point",
            x_offset_beam = x_offset_beam *10**-2,
            T_atoms=298.25, # Null out this contribution
            ###
            T_base=wire_no_beam.record_dict["T_distribution"][-1]

            ) 
        wire.simulate(n_steps=n_steps,
                        record_steps=record_steps,
                        time_step=time_step)
        


        run_name = "lw_{}_i_{}".format(l_wire,i_current)
        os.makedirs(plot_dir + "signal/", exist_ok=True)
        os.makedirs(plot_dir + "R_over_t/", exist_ok=True)
        wire.plot_signal(plot_dir + "signal/{}".format(run_name))
        wire.plot_R_over_t(plot_dir + "R_over_t/{}".format(run_name))
        os.makedirs(plot_dir + "heat_flow/", exist_ok=True)
        wire.plot_heat_flow(plot_dir + "heat_flow/{}".format(run_name))
        wire.plot_heat_flow(plot_dir + "heat_flow/{}_log".format(run_name),
                            log_y = True)

        wire.save(results_dir + "{}".format(run_name))

        time_after = time()
        run_time = time_after - time_before
        print("finished run: " + run_name + "time required: " 
                +  "{0:.2f} minutes".format(run_time/60))
        print("total time elapsed: {0:.2f} minutes".format(
                (time() - start_time)/60.0))
