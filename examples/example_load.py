# Add location of Wire_detector.py to path
import sys
import os

# top_dir = os.path.dirname(os.path.abspath(__file__)) + os.sep
# sys.path.append(top_dir + ".." + os.sep)
# import Wire class from Wire_detector.py 
from Wire_detector import Wire

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches

file_directory = "example/results/"
filename = "example_1"
wire = Wire().load(file_directory + filename)
os.makedirs("plots/", exist_ok=True)
#wire.plot_signal()

# wire.plot_signal() used as an example plot below
# Plot Temperature over Wire for start and end of simulation
# Plot Temperature over Wire
plt.figure(0, figsize=(8,6.5))
ax1 = plt.gca()
ax1.set_aspect(0.1)

x_lst = [1000 * ((i + 0.5) * wire.l_segment - (wire.l_wire / 2))
        for i in range(wire.n_wire_elements)]
T_beam_off = wire.record_dict["T_distribution"][0]
T_beam_on = wire.record_dict["T_distribution"][-1]
T_lst = [T_beam_off, T_beam_on]

R_arr = np.zeros(2)
for i,T_dist in enumerate(T_lst):
    wire.T_distribution = T_dist
    R_arr[i] = wire.resistance_total()

U_delta = (R_arr[1] - R_arr[0]) * wire.i_current
signal = (R_arr[1] - R_arr[0])/R_arr[0]

ax1.plot(x_lst, T_lst[0] - 273.15, "-", label=r"Beam Off, " 
            + "R = {:.3f}".format(R_arr[0]) + r"$\Omega$")
ax1.plot(x_lst, T_lst[1] - 273.15, "-", label=r"Beam On, " 
            + "R = {:.3f}".format(R_arr[1]) + r"$\Omega$")
            
ax1.set_ylabel("Temperature [°C]")
ax1.set_xlabel(r"wire positon [mm]")
plt.title(r"$d_{wire}$ = " + "{}".format(wire.d_wire * 10**6) 
            + r"$\mu m$" +", I = " + "{}".format(wire.i_current * 10**3)
            + r"$mA$" + r", $\phi_{beam}$ = 10^" + "{:.2f}".format(
            np.log10(wire.phi_beam)))
plt.grid(True)
# get existing handles and labels
handles, labels = plt.gca().get_legend_handles_labels()
# create a patch with no color
empty_patch = mpatches.Patch(color='none', label='Extra label') 
handles.append(empty_patch)
labels.append("Signal: {:.2%}, ".format(signal) + r"$\Delta U$" 
                + " = {:.2f}".format(U_delta *10 **3) + " mV, ")
plt.legend(handles, labels, shadow=True)


format_im = 'png' #'pdf' or png
dpi = 300
plt.savefig("plots/signal_plot" + '.{}'.format(format_im),
            format=format_im, dpi=dpi)
ax1.cla()

