import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as mpatches
import os
import dill
from wire_detector import Wire
from time import time

start_time = time()
top_dir = "2025_03_20_wire_l_sims_point/"
os.makedirs(top_dir, exist_ok=True)
results_dir = top_dir + "results/"
plot_dir = top_dir + "plots/"
os.makedirs(results_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)
d = 5
# i_current = 1
i_current = 0
# exp_list = np.linspace(14,18,num = 5)  # later normalized to per cm**2 
# exp_list = np.linspace(16,17,num = 1)  # later normalized to per cm**2 
# exp_list = np.linspace(17,18,num = 1)  # later normalized to per cm**2 
# exp_list = [16, 17, 0]
# exp_list = [0]
# exp_list = [1]
l_beam_list = [10, 1.6]  # in cm
# l_wire_list = [#5,4,3,
#                 #2.1,2.2,2.3,2.4,
#                #2.5
#                #2,
#                 1.7,1.9,
#                #1.5,1,
#                #0.5,0.2,0.1
#                ] # in cm

# l_wire_list = [5,4,3,
#                 2.1,2.2,2.3,2.4,
#                2.5,
#                2,
#                 1.7,1.9,
#                1.5,1,
#                0.5,0.2
#                ] # in cm

# l_wire_list = ([0.1 * i for i in range(2,20)] 
#              + [2 + 0.2 * i for i in range(0,26)]
#                )
l_wire_list = ([8 + 1 * i for i in range(0,8)]
               )
# #Choose every 5th for faster running
# l_wire_list = l_wire_list[::5]
# l_wire_list = [8]
l_wire_list = np.round(np.array(l_wire_list),decimals=1)
print("l_wire_list", l_wire_list)
min_segment_length = 0.2*10**-3
E_recombination = Wire().E_recombination

# power_watts_list = [100e-6, 1e-6]
power_watts_list = [1e-6, 100e-6]
beamshape_list = ["Point", "CenterPoint", "EveryPoint"]

for l_wire in l_wire_list:
    n_wire_elements = int(np.round(l_wire*10**-2 / min_segment_length))


    # if n_max > 100:
    #     n_wire_elements = 100
    # else:
    #     if round(n_max,-1) == 0:
    #         n_wire_elements = 10
    #     else:
    #         n_wire_elements = int(round(n_max,-1))
    # simulate wire base temperature with beam off
    wire_no_beam = Wire(
        n_wire_elements = n_wire_elements,
        i_current = (d/5)**2 * i_current * 10**-3, d_wire = d * 10**-6,
        emissivity = 0.3, l_wire=l_wire*10**-2,
        ###
        # T_base=None,
        T_cracker = 293.25, # Null out this contribution
        T_background=293.25,
        ### BEAM
        # use E_rec/2 factor to get input in Watts
        phi_beam= 0 / (E_recombination/2),
        beam_shape = "Point",
        ###
        ) 

    # Run the Simulation
    mod = 4

    n_steps_no_beam = 30000 * mod
    n_steps = 10000 * mod
    record_steps = 1000
    time_step = 0.001 / mod
    wire_no_beam.simulate(n_steps=n_steps_no_beam, record_steps=record_steps,
                            time_step=time_step)

    for beamshape in beamshape_list:
        for power_watts in power_watts_list:
            time_before = time()
            # simulate with beam on (Beam is wider than the wire length)
            wire = Wire(
                n_wire_elements = n_wire_elements,
                i_current = (d/5)**2 * i_current * 10**-3, d_wire = d * 10**-6,
                emissivity = 0.3, l_wire=l_wire*10**-2,
                ### BEAM
                # use E_rec/2 factor to get input in Watts
                phi_beam= power_watts / (E_recombination/2),
                beam_shape = beamshape,
                ###
                T_base=wire_no_beam.record_dict["T_distribution"][-1],
                #Remove f_bb to counter l_beam dependency of f_bb
                ###
                # T_base=None,
                T_cracker = 293.25, # Null out this contribution
                T_background=293.25,
                )

            wire.simulate(n_steps=n_steps, record_steps=record_steps,
                        time_step=time_step)

            run_name = "bs_{}_lw_{}_pow_{}_i_{}".format(
                beamshape, l_wire, power_watts,i_current)
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
