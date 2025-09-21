import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as mpatches
import os
import dill
from wire_detector import Wire
from time import time

start_time = time()
top_dir = "2025-08-27_sims_T_bkgd/"
os.makedirs(top_dir, exist_ok=True)
results_dir = top_dir + "results/"
plot_dir = top_dir + "plots/"
os.makedirs(results_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)
d = 5
# i_current = 1
i_current = 1
exp_list = np.linspace(17,18,num = 1) 
#l_beam_list = [10, 1.6]  # in cm


l_wire_list = ([2]
               )
T_bkgd_list = np.array([0,0.1, 1,2,10])+ 293.15

emissivity_list = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]

# #Choose every 5th for faster running
# l_wire_list = l_wire_list[::5]
# l_wire_list = [8]
l_wire_list = np.round(np.array(l_wire_list),decimals=1)
print("l_wire_list", l_wire_list)
min_segment_length = 0.2*10**-3
E_recombination = Wire().E_recombination

# beamshape_list = ["Tschersich"]
l_wire = 2
for eps in emissivity_list:
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
        emissivity = eps, l_wire=l_wire*10**-2,
        pressure = 2e-5,
        ###
        # T_base=None,
        A_cracker=0,
        T_cracker = 293.15, # Null out this contribution
        T_background=293.15,
        ### BEAM
        # use E_rec/2 factor to get input in Watts
        phi_beam= 0 / (E_recombination/2),
        beam_shape = "Point",
        ###
        ) 

    # Run the Simulation
    mod = 4
    #mod = 8

    n_steps_no_beam = 30000 * mod
    n_steps = 10000 * mod
    record_steps = 1000
    time_step = 0.001 / mod
    wire_no_beam.simulate(n_steps=n_steps_no_beam, record_steps=record_steps,
                            time_step=time_step)


    for T_bkgd in T_bkgd_list:
        time_before = time()
        # simulate with beam on (Beam is wider than the wire length)
        wire = Wire(
            n_wire_elements = n_wire_elements,
            i_current = (d/5)**2 * i_current * 10**-3, d_wire = d * 10**-6,
            emissivity = eps, l_wire=l_wire*10**-2,
            pressure = 2e-5,
            ### BEAM
            # use E_rec/2 factor to get input in Watts
            phi_beam= 0,
            # beam_shape = beamshape,
            ###
            T_base=wire_no_beam.record_dict["T_distribution"][-1],
            #Remove f_bb to counter l_beam dependency of f_bb
            ###
            # T_base=None,
            A_cracker=0,
            T_cracker = 293.15, # Null out this contribution
            T_background= 293.15,
            T_board= T_bkgd,
            )

        wire.simulate(n_steps=n_steps, record_steps=record_steps,
                    time_step=time_step)

        T_avg_before = np.average(
                    wire_no_beam.record_dict["T_distribution"][-1])
        T_avg_after = np.average(
                    wire.record_dict["T_distribution"][-1])
        print("T_avg_before:", 
              T_avg_before,
              "T_avg_after:", 
              T_avg_after,
              "Delta T_avg:", 
              T_avg_after - T_avg_before,
              )
        run_name = "T_bkgd_{:.2f}_eps_{:.2f}_i_{}".format(
            T_bkgd, eps, i_current)
        
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
