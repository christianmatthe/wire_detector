import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as mpatches
import os
import dill
from Wire_detector import Wire


fig = plt.figure(0, figsize=(8,6.5))
ax1=plt.gca()
top_dir_list = ["20um_phi_beam_sweep_taylor/", "20um_phi_beam_sweep/"]
#top_dir_list = ["10um_phi_beam_sweep_taylor/", "10um_phi_beam_sweep/"]
#top_dir_list = ["5um_phi_beam_sweep_taylor/", "5um_phi_beam_sweep/"]
label_list = [r"$F_{rad}$ Taylored", r"$F_{rad} \propto T^4$"]
exp_list = np.linspace(15,20,num = 21)
# Initialize Arrays
U_arr = np.zeros((len(top_dir_list), len(exp_list)))
T_max_arr = np.zeros((len(top_dir_list), len(exp_list)))
signal_arr = np.zeros((len(top_dir_list), len(exp_list)))
for j, top_dir in enumerate(top_dir_list):
    for i, phi_exp in enumerate(exp_list):
        wire = Wire()
        wire = wire.load(top_dir + "phi_to_{}".format(phi_exp))

        U_beam_off = wire.U_wire(0)
        U_beam_on = wire.U_wire(-1)
        
        U_delta = U_arr[j, i] = U_beam_on - U_beam_off
        signal = signal_arr[j, i] = U_delta / U_beam_off

        T_max = T_max_arr[j, i] = np.amax(wire.record_dict["T_distribution"]
                                          [-1])

    phi_list = 10**exp_list

# Plot delta U vs Phi in atoms/s
fig = plt.figure(0, figsize=(8,6.5))
ax1=plt.gca()

for j, top_dir in enumerate(top_dir_list):
    ax1.loglog(phi_list, U_arr[j]*1000, "-", label=label_list[j], basex=10)
ax1.set_ylabel(r"$\Delta U$ [mV]")
ax1.set_xlabel(r"$\Phi_{beam}$ [Atoms/s]")
plt.grid(True)
plt.legend(shadow=True)


format_im = 'png' #'pdf' or png
dpi = 300
plt.savefig(top_dir_list[0] + "plots/deltaU_compare" + '.{}'.format(format_im),
            format=format_im, dpi=dpi)
ax1.cla()

# Plot delta U vs Phi in sccm
fig = plt.figure(0, figsize=(8,6.5))
ax1=plt.gca()
sccm_convert = 4.478 * 10**17
for j, top_dir in enumerate(top_dir_list):
    ax1.loglog(phi_list/sccm_convert, U_arr[j]*1000, "-", label=label_list[j],
            basex=10)
ax1.set_ylabel(r"$\Delta U$ [mV]")
ax1.set_xlabel(r"$\Phi_{beam}$ [sccm]")
plt.grid(True)
plt.legend(shadow=True)


format_im = 'png' #'pdf' or png
dpi = 300
plt.savefig(top_dir_list[0] + "plots/deltaU_sccm_compare" + '.{}'.format(
            format_im), format=format_im, dpi=dpi)
ax1.cla()

# Plot delta U vs T_max
fig = plt.figure(0, figsize=(8,6.5))
ax1=plt.gca()
for j, top_dir in enumerate(top_dir_list):
    ax1.loglog(T_max_arr[j] - T_max_arr[j][0], U_arr[j]*1000, "-x", label=label_list[j],
               basex=10)
ax1.set_ylabel(r"$\Delta U$ [mV]")
ax1.set_xlabel(r"$\Delta$ Temperature [K]")
plt.grid(True)
plt.legend(shadow=True)


format_im = 'png' #'pdf' or png
dpi = 300
plt.savefig(top_dir_list[0] + "plots/deltaU_T_compare" + '.{}'.format(
            format_im),format=format_im, dpi=dpi)
ax1.cla()

# Plot T_max vs Phi in Atoms per second
fig = plt.figure(0, figsize=(8,6.5))
ax1=plt.gca()
for j, top_dir in enumerate(top_dir_list):
    ax1.loglog(phi_list, T_max_arr[j] - T_max_arr[j][0] , "-", label=label_list[j],
               basex=10)
ax1.set_ylabel(r"$\Delta$ Temperature [K]")
ax1.set_xlabel(r"$\Phi_{beam}$ [Atoms/s]")
plt.grid(True)
plt.legend(shadow=True)


format_im = 'png' #'pdf' or png
dpi = 300
plt.savefig(top_dir_list[0] + "plots/phi_T_compare" + '.{}'.format(
            format_im),format=format_im, dpi=dpi)
ax1.cla()

# Plot T_max vs Phi in Atoms per second
fig = plt.figure(0, figsize=(8,6.5))
ax1=plt.gca()
for j, top_dir in enumerate(top_dir_list):
    ax1.loglog(phi_list/sccm_convert, T_max_arr[j] - T_max_arr[j][0] , "-", label=label_list[j],
               basex=10)
ax1.set_ylabel(r"$\Delta$ Temperature [K]")
ax1.set_xlabel(r"$\Phi_{beam}$ [sccm]")
plt.grid(True)
plt.legend(shadow=True)


format_im = 'png' #'pdf' or png
dpi = 300
plt.savefig(top_dir_list[0] + "plots/phi_sccm_T_compare" + '.{}'.format(
            format_im),format=format_im, dpi=dpi)
ax1.cla()