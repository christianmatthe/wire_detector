# Add location of Wire_detector.py to path
import sys
sys.path.append("..")

# import Wire class from Wire_detector.py 
from Wire_detector import Wire

import os

print("This should take about 3 minutes to run")
# initialize wire with selected parameters
wire_no_beam = Wire(
            n_wire_elements = 100,
            i_current = 1 * 10**-3, d_wire = 5 * 10**-6,
            emissivity = 0.3, l_wire= 2.7 *10**-2,
            ### Without beam and with default stating temperature distribution
            phi_beam=0, T_base=None
            ###
            )

# Run Simulation to heat wire to starting temperature distribution
# adjust mod for shorter time steps if simulation crashes due to overflows
mod = 2
n_steps_no_beam = 20000 * mod
n_steps = 10000 * mod
record_steps = 1000
time_step = 0.001 / mod
wire_no_beam.simulate(n_steps=n_steps_no_beam, 
                        record_steps=record_steps, time_step=time_step)

# clone wire object
wire = wire_no_beam
# Adjust parameters to turn  beam on
wire.phi_beam = 10**17
wire.beam_shape = "Flat"
wire.l_beam = 1.6 * 10**-2
#set base temperature distribution to end state of no beam simulation
wire.T_base = wire_no_beam.record_dict["T_distribution"][-1]

# simulate wire with beam
wire.simulate(n_steps=n_steps, record_steps=record_steps,
              time_step=time_step)

# Save some plots showing results
top_dir = "example/"
os.makedirs(top_dir, exist_ok=True)
results_dir = top_dir + "results/"
plot_dir = top_dir + "plots/"
os.makedirs(results_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)


run_name = "example_{}".format("1")
os.makedirs(plot_dir + "signal/", exist_ok=True)
os.makedirs(plot_dir + "R_over_t/", exist_ok=True)
wire.plot_signal(plot_dir + "signal/{}".format(run_name))
wire.plot_R_over_t(plot_dir + "R_over_t/{}".format(run_name))
os.makedirs(plot_dir + "heat_flow/", exist_ok=True)
wire.plot_heat_flow(plot_dir + "heat_flow/{}".format(run_name))
wire.plot_heat_flow(plot_dir + "heat_flow/log_{}".format(
                    run_name), log_y =True)

# Save wire object for future use
wire.save(results_dir + "{}".format(run_name))