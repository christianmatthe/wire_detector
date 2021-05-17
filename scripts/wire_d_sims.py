import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as mpatches
import os
import dill
from Wire_detector import Wire
from time import time

start_time = time()
# Simulate different wire thicknesses with base voltage matched using current
# over range of beam Flat strengths
top_dir = "wire_d_sims/"
os.makedirs(top_dir, exist_ok=True)
results_dir = top_dir + "results/"
plot_dir = top_dir + "plots/"
os.makedirs(results_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)
d_wire_list = [5,10,20]
i_current_list = [0.1, 1]
exp_list = np.linspace(14,20,num = 25)  # later normalized to per cm**2 
l_beam = 10*10**-2
A_beam = np.pi * (l_beam/2)**2

for i_current in i_current_list:
    for d in d_wire_list:
        # simulate wire base temperature with beam off
        wire_no_beam = Wire(
            n_wire_elements = 100, k_heat_conductivity = 174,
            i_current = (d/5)**2 * i_current * 10**-3, d_wire = d * 10**-6,
            emissivity = 0.3, l_wire=5.45*10**-2,
            beam_shape="Flat", l_beam = l_beam,
            ###
            phi_beam=0, T_base=None
            ###
            ) 

        # Run the Simulation
        n_steps = 20000
        record_steps = 1000
        time_step = 0.001
        wire_no_beam.simulate(n_steps=n_steps, record_steps=record_steps,
                                time_step=time_step)
        for phi_exp in exp_list:
            time_before = time()
            # simulate with beam on (Beam is wider than the wire length)
            wire = Wire(
                n_wire_elements = 100, k_heat_conductivity = 174,
                i_current = (d/5)**2 * i_current * 10**-3, d_wire = d * 10**-6,
                emissivity = 0.3, l_wire=5.45*10**-2,
                beam_shape="Flat", l_beam = l_beam, 
                phi_beam= (A_beam/ 10**-4) * 10**phi_exp, # Normalized to cm**2
                T_base=wire_no_beam.record_dict["T_distribution"][-1]
                )

            wire.simulate(n_steps=n_steps, record_steps=record_steps,
                        time_step=time_step)

            run_name = "d_{}_i_{}_phi_{}".format(d, i_current, phi_exp)
            os.makedirs(plot_dir + "signal/", exist_ok=True)
            os.makedirs(plot_dir + "R_over_t/", exist_ok=True)
            wire.plot_signal(plot_dir + "signal/{}".format(run_name))
            wire.plot_R_over_t(plot_dir + "R_over_t/{}".format(run_name))
            wire.save(results_dir + "{}".format(run_name))

            time_after = time()
            run_time = time_after - time_before
            print("finished run: " + run_name + "time required: " 
                  +  "{0:.2f} minutes".format(run_time/60))
            print("total time elapsed: {0:.2f} minutes".format(
                  (time() - start_time)/60.0))
