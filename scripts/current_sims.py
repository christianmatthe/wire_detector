import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as mpatches
import os
import dill
from Wire_detector import Wire
from time import time

start_time = time()
top_dir = "current_sims_2/0.03mbar_air/"
os.makedirs(top_dir, exist_ok=True)
results_dir = top_dir + "results/"
plot_dir = top_dir + "plots/"
os.makedirs(results_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)
d = 5
i_current_list = [0.5,0.7,1,1.2,1.5,2,3,4,5,6,7,8,9,10]
x_offset_list = np.linspace(0,2.5,num = 26)
exp_list = [17]
#exp_list = np.linspace(14,18,num = 5)  # later normalized to per cm**2 
l_beam_list = [1.6]  # in cm
l_wire_list = [#5,
                2.7
               ] # in cm

min_segment_length = 0.1*10**-3
for l_wire in l_wire_list:
    n_max = l_wire*10**-2 / min_segment_length
    if n_max > 100:
        n_wire_elements = 100
    else:
        if round(n_max,-1) == 0:
            n_wire_elements = 10
        else:
            n_wire_elements = int(round(n_max,-1))
    # simulate wire base temperature with beam off

    for i_current in i_current_list:
        time_before = time()
        wire_no_beam = Wire(
            n_wire_elements = n_wire_elements,
            i_current = (d/5)**2 * i_current * 10**-3, d_wire = d * 10**-6,
            emissivity = 0.3, l_wire=l_wire*10**-2, 
            rho_specific_resistance=0.054 * 10**-6,
            pressure=0.03*10**-3*10**5, m_molecular_gas= 29 * 1.674 * 10**-27,
            T_cracker = 300,
            ###
            phi_beam=0, T_base=None
            ###
            ) 

        # Run the Simulation
        mod = 4

        n_steps_no_beam = 30000 * mod
        n_steps = 10000 * mod
        record_steps = 1000
        time_step = 0.001 / mod
        wire_no_beam.simulate(n_steps=n_steps_no_beam,
                              record_steps=record_steps,
                              time_step=time_step)

        wire = wire_no_beam
        run_name = "lw_{}_i_{}".format(l_wire,i_current)
        os.makedirs(plot_dir + "signal/", exist_ok=True)
        os.makedirs(plot_dir + "R_over_t/", exist_ok=True)
        wire.plot_signal(plot_dir + "signal/{}".format(run_name))
        wire.plot_R_over_t(plot_dir + "R_over_t/{}".format(run_name))
        os.makedirs(plot_dir + "heat_flow/", exist_ok=True)
        wire.plot_heat_flow(plot_dir + "heat_flow/{}".format(run_name))

        wire.save(results_dir + "{}".format(run_name))

        time_after = time()
        run_time = time_after - time_before
        print("finished run: " + run_name + "time required: " 
                +  "{0:.2f} minutes".format(run_time/60))
        print("total time elapsed: {0:.2f} minutes".format(
                (time() - start_time)/60.0))