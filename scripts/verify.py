# Verify values form 1996 paper
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as mpatches
import os
import dill
from Wire_detector import Wire

######################
# top_dir = "verify/"
# os.makedirs(top_dir, exist_ok=True)
# os.makedirs(top_dir + "plots/", exist_ok=True)
# # Alexandre 96 Paper values
# d_wire_list = [2, 3, 4, 5, 6, 7, 8, 9, 10]
# i_current = 0.1 * 10**-3
# for d in d_wire_list:
#     # simulate wire base temperature with beam off
#     wire_no_beam = Wire(
#         n_wire_elements = 100, k_heat_conductivity = 170,
#         i_current = i_current, d_wire = d * 10**-6,
#         emissivity = 0.3, l_wire=5*10**-2,
#         rho_specific_resistance = 668 * (np.pi*((10/2)*10**-6)**2),
#         ###
#         phi_beam=0, T_base=None
#         ###
#         )


#     # Run the Simulation
#     n_steps = 30000
#     record_steps = 150
#     time_step = 0.001
#     wire_no_beam.simulate(n_steps=n_steps, record_steps=record_steps,
#                           time_step=time_step)


#     # simulate with beam on
#     wire = Wire(n_wire_elements = 100, k_heat_conductivity = 170,
#                 i_current = i_current, d_wire = d * 10**-6,
#                 emissivity = 0.3, l_wire=5*10**-2, 
#                 rho_specific_resistance = 668 * (np.pi*((10/2)*10**-6)**2),
#                 beam_shape="Flat",
#                 T_base=wire_no_beam.record_dict["T_distribution"][-1]
#     )

#     wire.simulate(n_steps=n_steps, record_steps=record_steps,
#                 time_step=time_step)
#     wire.plot_signal(top_dir + "plots/d_wire_{}".format(d))
#     wire.plot_R_over_t(top_dir + "plots/R_over_t_{}um".format(d))
#     wire.save(top_dir + "d_wire_{}".format(d)) 
#############################

# # Imitate Values from AG Pohl
# top_dir = "AG_Pohl_3/"
# os.makedirs(top_dir, exist_ok=True)
# os.makedirs(top_dir + "plots/", exist_ok=True)
# d_wire_list = [5, 10, 20]
# i_current = 1 * 10**-3
# for d in d_wire_list:
#     # simulate wire base temperature with beam off
#     wire_no_beam = Wire(n_wire_elements = 100, k_heat_conductivity = 174,
#                         i_current = i_current, d_wire = d * 10**-6,
#                         emissivity = 0.3, l_wire=5.45*10**-2,
#                         beam_shape="Gaussian", sigma_beam=1.1*10**-3, 
#                         ###
#                         phi_beam=0, T_base=None
#                         ###
#             ) 


#     # Run the Simulation
#     n_steps = 20000
#     record_steps = 1000
#     time_step = 0.001
#     wire_no_beam.simulate(n_steps=n_steps, record_steps=record_steps,
#                           time_step=time_step)


#     # simulate with beam on
#     wire = Wire(n_wire_elements = 100, k_heat_conductivity = 174,
#                         i_current = i_current, d_wire = d * 10**-6,
#                         emissivity = 0.3, l_wire=5.45*10**-2,
#                         beam_shape="Gaussian", sigma_beam=1.1*10**-3, 
#                         phi_beam=10**17,
#                         T_base=wire_no_beam.record_dict["T_distribution"][-1]
#     )

#     wire.simulate(n_steps=n_steps, record_steps=record_steps,
#                 time_step=time_step)
#     wire.plot_signal(top_dir + "plots/d_wire_{}".format(d))
#     wire.plot_R_over_t(top_dir + "plots/R_over_t_{}um".format(d))
#     wire.save(top_dir + "d_wire_{}".format(d))
######################

# Try matching 5um signa with 20um wire by increasing current
top_dir = "low_current_matching/"
os.makedirs(top_dir, exist_ok=True)
os.makedirs(top_dir + "plots/", exist_ok=True)
d_wire_list = [5,10,20]
i_current = 0.1 * 10**-3
for d in d_wire_list:
    # simulate wire base temperature with beam off
    wire_no_beam = Wire(n_wire_elements = 100, k_heat_conductivity = 174,
                        i_current = (d/5)**2 * i_current, d_wire = d * 10**-6,
                        emissivity = 0.3, l_wire=5.45*10**-2,
                        beam_shape="Gaussian", sigma_beam=1.1*10**-3, 
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


    # simulate with beam on
    wire = Wire(n_wire_elements = 100, k_heat_conductivity = 174,
                        i_current = (d/5)**2 * i_current, d_wire = d * 10**-6,
                        emissivity = 0.3, l_wire=5.45*10**-2,
                        beam_shape="Gaussian", sigma_beam=6*10**-3, 
                        phi_beam=10**16,
                        T_base=wire_no_beam.record_dict["T_distribution"][-1]
    )

    wire.simulate(n_steps=n_steps, record_steps=record_steps,
                time_step=time_step)
    wire.plot_signal(top_dir + "plots/d_wire_{}".format(d))
    wire.plot_R_over_t(top_dir + "plots/R_over_t_{}um".format(d))
    wire.save(top_dir + "d_wire_{}".format(d))