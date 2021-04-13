import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as mpatches
import os
import dill
from Wire_detector import Wire

top_dir = os.path.dirname(os.path.abspath(__file__)) + "\\"

wire = Wire()
wire = wire.load("10um_phi_beam_sweep/" + "phi_to_17.0")
wire.taylor_f_rad = False
animate_dir = top_dir + "animate\\"
os.makedirs(animate_dir, exist_ok=True)

n_steps = 15000
record_steps = 150
time_step = 0.001
# ####### reproduce ohmic heating procedure
# wire_no_beam = wire
# wire_no_beam.T_base = None
# wire_no_beam.phi_beam = 0
# wire_no_beam.taylor_f_rad = False

# wire_no_beam.simulate(n_steps=n_steps, record_steps=record_steps,
#                      time_step=time_step)
# wire_no_beam.save(animate_dir + "base_heating")
# #######
wire_no_beam = Wire()
wire_no_beam = wire_no_beam.load(animate_dir + "base_heating")
plot_arr_0 = wire_no_beam.record_dict["T_distribution"] - 273.15
time_arr_0 = wire_no_beam.record_dict["time"]

x_lst = [1000 * ((i + 0.5) * wire.l_segment - (wire.l_wire / 2))
             for i in range(wire.n_wire_elements)]
plot_arr = wire.record_dict["T_distribution"] - 273.15
time_arr = wire.record_dict["time"]

# Animation
# First set up the figure, the axis, and the plot element we want to
# animate
fig = plt.figure(figsize=(8,6.5))
margin = 0.05
y_min = min([min(plot_arr[i]) for i in range(len(plot_arr))])
y_max = max([max(plot_arr[i]) for i in range(len(plot_arr))])
ylims = (y_min - margin * (y_max - y_min), y_max + margin * (y_max - y_min))
x_min = x_lst[0]
x_max = x_lst[-1]
xlims = (x_min - margin * (x_max - x_min), x_max + margin * (x_max - x_min))
ax = plt.axes(xlim=xlims, ylim=ylims)
line, = ax.plot([], [], lw=2)
time_text = ax.text(x_max*0.82, y_max*0.9, '0s', fontsize=15)
ax.set_ylabel("Temperature [°C]")
ax.set_xlabel(r"wire positon [mm]")


# initialization function: plot the background of each frame
def init():
    #line.set_data([], [])
    line.set_data(x_lst, plot_arr[-1])
    #line, = ax.plot(x_lst, plot_arr[-1], lw = 2)
    #line.set_xdata(x_lst)
    #line.set_ydata(plot_arr[-1])
    return line,

# animation function. This is called sequentially
def animate(frame):
    if frame < record_steps:
        y = plot_arr_0[frame]
        line.set_ydata(y)
        time_text.set_text("{:.1f}s".format(time_arr_0[frame]))
    else:
        y = plot_arr[frame - record_steps]
        line.set_ydata(y)
        time_text.set_text("{:.1f}s".format(time_arr[frame - record_steps] 
                                            + time_arr_0[-1]))

    return line,time_text

frames = [i for i in range(2*record_steps)]
ani = FuncAnimation(fig, animate, init_func=init, frames=frames,
                    interval=time_step*1000*(n_steps // record_steps),
                    repeat_delay=0
                    )
plt.grid(True)
ani.save(animate_dir + 'basic_animation.gif', writer = "imagemagick", fps=30
        )
#plt.show()

# Calculate heat flow over time
x_lst = [1000 * ((i + 0.5) * wire.l_segment - (wire.l_wire / 2))
            for i in range(wire.n_wire_elements)]

t_lst = wire.record_dict["time"]
f_el_arr = np.zeros((record_steps + 1, wire.n_wire_elements))
f_conduction_arr = np.zeros((record_steps + 1, wire.n_wire_elements))
f_rad_arr = np.zeros((record_steps + 1, wire.n_wire_elements))
f_beam_arr = np.zeros((record_steps + 1, wire.n_wire_elements))
for i,time in enumerate(wire.record_dict["time"]):
    wire.T_distribution = wire.record_dict["T_distribution"][i]
    f_el_arr[i] = [wire.f_el(j) for j in range(wire.n_wire_elements)]
    f_conduction_arr[i] = [wire.f_conduction(j) 
                            for j in range(wire.n_wire_elements)]
    f_rad_arr[i] = [wire.f_rad(j) for j in range(wire.n_wire_elements)]
    f_beam_arr[i] = [wire.f_beam(j) for j in range(wire.n_wire_elements)]

#Heat flow in ohmic heating lead up
t_lst_0 = wire_no_beam.record_dict["time"]
f_el_arr_0 = np.zeros((record_steps + 1, wire.n_wire_elements))
f_conduction_arr_0 = np.zeros((record_steps + 1, wire.n_wire_elements))
f_rad_arr_0 = np.zeros((record_steps + 1, wire.n_wire_elements))
f_beam_arr_0 = np.zeros((record_steps + 1, wire.n_wire_elements))
for i,time in enumerate(wire_no_beam.record_dict["time"]):
    wire_no_beam.T_distribution = wire_no_beam.record_dict["T_distribution"][i]
    f_el_arr_0[i] = [wire_no_beam.f_el(j) for j in range(wire.n_wire_elements)]
    f_conduction_arr_0[i] = [wire_no_beam.f_conduction(j) 
                            for j in range(wire.n_wire_elements)]
    f_rad_arr_0[i] = [wire_no_beam.f_rad(j) 
                      for j in range(wire.n_wire_elements)]
    f_beam_arr_0[i] = [wire_no_beam.f_beam(j) 
                       for j in range(wire.n_wire_elements)]

# Animation
# First set up the figure, the axis, and the plot element we want to
# animate
fig = plt.figure(figsize=(8,6.5))
margin = 0.05
y_min = min([min([min(f_el_arr[i]), min(f_conduction_arr[i]), min(f_rad_arr[i])
            , min(f_beam_arr[i])])
            for i in range(len(f_el_arr))])
y_max = max([max([max(f_el_arr[i]), max(f_conduction_arr[i]), max(f_rad_arr[i])
            , max(f_beam_arr[i])])
            for i in range(len(f_el_arr))])
ylims = (y_min - margin * (y_max - y_min), y_max + margin * (y_max - y_min))
x_min = x_lst[0]
x_max = x_lst[-1]
xlims = (x_min - margin * (x_max - x_min), x_max + margin * (x_max - x_min))
ax = plt.axes(xlim=xlims, ylim=ylims)
line_el, = ax.plot([], [], lw=2, label=r"$F_{el}$")
line_con, = ax.plot([], [], lw=2, label=r"$F_{conduction}$")
line_rad, = ax.plot([], [], lw=2, label=r"$F_{rad}$")
line_beam, = ax.plot([], [], lw=2, label=r"$F_{beam}$")
time_text = ax.text(x_max*0.82, y_max*0.9, '0s', fontsize=15)
ax.set_ylabel("Heat Flow [W/m]")
ax.set_xlabel(r"wire positon [mm]")

# initialization function: plot the background of each frame
def init_heat():
    #line.set_data([], [])
    line_el.set_data(x_lst, f_el_arr[-1])
    line_con.set_data(x_lst, f_conduction_arr[-1])
    line_rad.set_data(x_lst, f_rad_arr[-1])
    line_beam.set_data(x_lst, f_beam_arr[-1])
    return line_el, line_con, line_rad, line_beam 

# animation function. This is called sequentially
def animate_heat(frame):
    # if frame < record_steps:
    #     y = plot_arr_0[frame]
    #     line.set_ydata(y)
    #     time_text.set_text("{:.1f}s".format(time_arr_0[frame]))
    # else:
    #     y = plot_arr[frame - record_steps]
    #     line.set_ydata(y)
    #     time_text.set_text("{:.1f}s".format(time_arr[frame - record_steps] 
    #                                         + time_arr_0[-1]))
    #y = f_el_arr[frame]
    if frame < record_steps:
        line_el.set_ydata(f_el_arr_0[frame])
        line_con.set_ydata(f_conduction_arr_0[frame])
        line_rad.set_ydata(f_rad_arr_0[frame])
        line_beam.set_ydata(f_beam_arr_0[frame])

        time_text.set_text("{:.1f}s".format(t_lst[frame]))
    else:
        line_el.set_ydata(f_el_arr[frame - record_steps])
        line_con.set_ydata(f_conduction_arr[frame - record_steps])
        line_rad.set_ydata(f_rad_arr[frame - record_steps])
        line_beam.set_ydata(f_beam_arr[frame - record_steps])

        time_text.set_text("{:.1f}s".format(t_lst[frame - record_steps]
                                            + t_lst_0[-1]))

    return line_el, line_con, line_rad, line_beam ,time_text

frames = [i for i in range(2*record_steps)]
ani = FuncAnimation(fig, animate_heat, init_func=init_heat, frames=frames,
                    interval=time_step*1000*(n_steps // record_steps),
                    repeat_delay=0
                    )
plt.grid(True)
plt.legend(shadow=True, loc = "upper left")
ani.save(animate_dir + 'heat_flow.gif', writer = "imagemagick", fps=30
        )

# # Combined animation:
# fig = plt.figure(0, figsize=(12,6.5))

# gs = mpl.gridspec.GridSpec(2, 1) 
# gs.update(wspace=0.05)

# ax1 = plt.subplot(gs[0])
# ax2 = plt.subplot(gs[1])

# import imageio
# import numpy as np    

# #Create reader object for the gif
# gif1 = imageio.get_reader(animate_dir + 'basic_animation.gif')
# gif2 = imageio.get_reader(animate_dir + 'heat_flow.gif')

# #If they don't have the same number of frame take the shorter
# number_of_frames = min(gif1.get_length(), gif2.get_length()) 

# #Create writer object
# new_gif = imageio.get_writer(animate_dir + 'combined.gif')

# for frame_number in range(number_of_frames):
#     img1 = gif1.get_next_data()
#     img2 = gif2.get_next_data()
#     #here is the magic
#     new_image = np.hstack((img1, img2))
#     new_gif.append_data(new_image)

# gif1.close()
# gif2.close()    
# new_gif.close()

