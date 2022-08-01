from numpy.lib.shape_base import take_along_axis
import pandas as pd
import datetime as dt
import matplotlib.pyplot  as plt
import numpy as np
from scipy.optimize import curve_fit

#plot Options
import matplotlib as mpl
font = {#'family' : 'normal','weight' : 'bold',
        'size'   : 16
        #,'serif':['Helvetica']
        }
mpl.rc('font', **font)

def resistance(U, I):
    return U/I

def t_diff(start_time, end_time):
    return (end_time - start_time).total_seconds()

def t_diff_list(data_frame, index_list):
    start_index = index_list[0]
    lst = [t_diff(data_frame["date_time"][start_index],
                  data_frame["date_time"][i]) for i in index_list]
    return lst



def make_datetime(data_frame):
    data_frame["date_time"] = [dt.datetime(1,1,1) for i in range(len(data_frame))]
    for index, row in data_frame.iterrows():
        date_list = data_frame["Date"][index].split("-")
        time_list = data_frame["Time"][index].split(":")
        date_time =dt.datetime(year = int(date_list[2]),
                               month = int(date_list[1]),
                               day = int(date_list[0]),
                               hour = int(time_list[0]),
                               minute = int(time_list[1])   )
        #print(date_time)
        data_frame["date_time"][index] = date_time

# setup the fire directory
file_dir = "C:\\Users\\Christian\\Documents\\StudiumPhD\\Measurements\\"

file = file_dir + '2021_06_29_tungsten_ablation_test.csv'
data_frame = pd.read_csv(file)

make_datetime(data_frame)

U_list = data_frame["U meas (mV)"].values.tolist()
I_list = data_frame["I meas (mA)"].values.tolist()

index_lowP = [i for i in range(4,15)] + [i for i in range(21,27)]
#print(index_lowP)

index_highP = [i  for i  in range(32,37)]
#print(index_highP)

R_list = [resistance(U_list[i],I_list[i]) for i  in index_lowP]
#print(R_list)
t_list = np.array(t_diff_list(data_frame, index_lowP))/3600
#print(t_list)

R_highP_list = [resistance(U_list[i],I_list[i]) for i  in index_highP]
t_highP_list = np.array(t_diff_list(data_frame, index_highP))/3600


#define fit function
def fit_func(x,m,b):
    return m*x + b

x_list = np.array(t_list)
popt, pcov = curve_fit(fit_func, x_list, R_list)

fig = plt.figure(0, figsize=(8,6.5))
ax1=plt.gca()
ax1.plot(x_list, R_list, ".")
ax1.plot(x_list, fit_func(x_list, *popt), 'r-',
         label='fit: m=%5.3f, b=%5.3f' % tuple(popt))

ax1.set_xlabel("t [h]")
ax1.set_ylabel(r"R [$\Omega$]")
plt.grid(True)
plt.legend(shadow=True)

format_im = 'png' #'pdf' or png
dpi = 300
plt.savefig("ablation"
            + '.{}'.format(format_im),
            format=format_im, dpi=dpi)
ax1.cla()

# repeat for high pressures
x_list = np.array(t_highP_list)
popt, pcov = curve_fit(fit_func, x_list, R_highP_list )

fig = plt.figure(0, figsize=(8,6.5))
ax1=plt.gca()
ax1.plot(x_list, R_highP_list, ".")
ax1.plot(x_list, fit_func(x_list, *popt), 'r-',
         label='fit: m=%5.3f, b=%5.3f' % tuple(popt))

ax1.set_xlabel("t [h]")
ax1.set_ylabel(r"R [$\Omega$]")
plt.grid(True)
plt.legend(shadow=True)

format_im = 'png' #'pdf' or png
dpi = 300
plt.savefig("ablation_high"
            + '.{}'.format(format_im),
            format=format_im, dpi=dpi)
ax1.cla()