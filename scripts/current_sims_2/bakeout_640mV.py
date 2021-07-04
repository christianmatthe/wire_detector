import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
# 0.04mbar run, Scroll pump
file_dir = "C:\\Users\\Christian\\Documents\\StudiumPhD\\Measurements\\"
# file = file_dir + '2021-06-07_wire_3_post_300C_bakeout_fill.csv'
# data_frame = pd.read_csv(file)

# df = data_frame[6:64]

file = file_dir + '2021-05-10_test_vac_0.04_mbar_fill.csv'
data_frame = pd.read_csv(file)

df_bakeout = data_frame[119:164]
df_bakeout = data_frame[119:164]
#print(df_bakeout)

def make_datetime(data_frame):
    data_frame["date_time"] = [dt.datetime(1,1,1) for i in range(len(data_frame))]
    for index, row in df_bakeout.iterrows():
        date_list = data_frame["Date"][index].split("-")
        time_list = data_frame["Time"][index].split(":")
        date_time =dt.datetime(year = int(date_list[2]),
                               month = int(date_list[1]),
                               day = int(date_list[0]),
                               hour = int(time_list[0]),
                               minute = int(time_list[1])   )
        #print(date_time)
        data_frame["date_time"][index] = date_time

make_datetime(df_bakeout)
#print(df_bakeout)
# for i  in range(119,163):
#     a = df_bakeout["date_time"][i] - df_bakeout["date_time"][119]
#     print(a)
#     b = a.total_seconds()/60
#     print(b)
#     #print(b)
import matplotlib as mpl
font = {#'family' : 'normal','weight' : 'bold',
        'size'   : 16
        #,'serif':['Helvetica']
        }
mpl.rc('font', **font)

t_delta_list = [(df_bakeout["date_time"][i] - df_bakeout["date_time"][119]
                ).total_seconds()/60
                for i in range(119,164)]
print(t_delta_list)
fig = plt.figure(0, figsize=(8,6.5))
ax1=plt.gca()
ax1.errorbar(t_delta_list, df_bakeout["I meas (mA)"], df_bakeout["err I"], 
             marker = ".", linestyle="None")
ax1.set_ylabel(r"I [mA]")
ax1.set_xlabel(r"t [min]")
plt.grid(True)
#plt.legend(shadow=True)

format_im = 'png' #'pdf' or png
dpi = 300
plt.savefig("bakeout_i_over_t"
            + '.{}'.format(format_im),
            format=format_im, dpi=dpi)
ax1.cla()

fig = plt.figure(0, figsize=(8,6.5))
ax1=plt.gca()
ax1.plot(t_delta_list, df_bakeout["T (°C)"],
             marker = ".", linestyle="None")
ax1.set_ylabel(r"T [°C]")
ax1.set_xlabel(r"t [min]")
plt.grid(True)
#plt.legend(shadow=True)

format_im = 'png' #'pdf' or png
dpi = 300
plt.savefig("bakeout_T_over_t"
            + '.{}'.format(format_im),
            format=format_im, dpi=dpi)
ax1.cla()


#stability after backkeout
df_after = data_frame[164:167]

make_datetime(df_after)

t_delta_list = [(df_after["date_time"][i] - df_after["date_time"][164]
                ).total_seconds()/60
                for i in range(164,166)]
print(t_delta_list)
fig = plt.figure(0, figsize=(8,6.5))
ax1=plt.gca()
ax1.errorbar(t_delta_list, df_after["I meas (mA)"], yerr = df_after["err I"], 
             marker = ".", linestyle="None")
ax1.set_ylabel(r"I [mA]")
ax1.set_xlabel(r"t [min]")
plt.grid(True)
#plt.legend(shadow=True)

format_im = 'png' #'pdf' or png
dpi = 300
plt.savefig("test_after"
            + '.{}'.format(format_im),
            format=format_im, dpi=dpi)
ax1.cla()