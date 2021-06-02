import numpy as np
import matplotlib.pyplot as plt
from Wire_detector import Wire
import os
from numpy import sum
import scipy
import sys
import pandas as pd
power = pd.read_csv('2021-05-10_test_vac_0.04_mbar_fill.csv')
conductivity=pd.read_csv('Thermal-conductivity-of-Tungsten.csv')
from scipy.interpolate import interp1d
from scipy import interpolate

wire= Wire(
T_background = 293.15,
l_wire= 2.7 *10**-2,
d_wire = 5 * 10**-6,
l_beam = 26 * 10**-3,
emissivity=0.25,
n_wire_elements=900
)
i1=[i*10**-5 for i in range(1,9,3)]
i2=[i*10**-5 for i in range(10,90,10)]
i3=[i*10**-5 for i in range(100,200,5)]
i_current_list= i1+i2+i3
phi_beam =  (10**(17.73))
R_0=wire.rho_specific_resistance/wire.A_cross_section
C=2*phi_beam*wire.d_wire*wire.E_recombination/(np.pi*wire.l_beam**2)
R_base=R_0*wire.l_wire

L1=np.linspace(-wire.l_wire/2, -wire.l_beam/2,300)
L2=np.linspace(-wire.l_beam/2,wire.l_beam/2,300)
L3=np.linspace(wire.l_beam/2,wire.l_wire/2,300)
L=np.concatenate((L1,L2,L3))



P_radiation=[]
P_electrique=[]
P_conduction=[]
P_backgroundgaz=[]
T_av=[]
for i in i_current_list:
    beta1=C+R_0*((i)**2)/(wire.k_heat_conductivity*wire.A_cross_section)
    beta2=R_0*((i)**2)/(wire.k_heat_conductivity*wire.A_cross_section)
    A=((4*np.pi*wire.sigma_stefan_boltzmann*wire.emissivity*wire.d_wire*wire.T_background**3)-(i**2)*R_0*wire.a_temperature_coefficient)/(wire.k_heat_conductivity*wire.A_cross_section)
    x_0=np.lib.scimath.sqrt(A)*wire.l_wire/2
    x_1=(np.lib.scimath.sqrt(A))*wire.l_beam/2
    y_0=-beta2/A
    y_1=-beta1/A
    R_beamOFF= R_base + R_0*wire.a_temperature_coefficient* (- y_0*wire.l_wire+ (2*y_0*np.sinh( x_0 ) / ( np.lib.scimath.sqrt(A)*np.cosh( x_0 ))))
    def C1(y_0,y_1):
        C1=( y_0 + ( y_0 - y_1 )*np.sinh( x_1 )*np.sinh( x_0 ))/np.cosh( x_0 )
        return C1
    def C2(y_0,y_1):
        C2=( y_0 - y_1 )*np.sinh( x_1)
        return C2
    def C5(y_0,y_1):
        C5 = ( y_0 + ( y_0 - y_1 )*(np.sinh( x_1 )*np.sinh( x_0 )-np.cosh(x_1)*np.cosh(x_0)))/np.cosh( x_0 )
        return C5
    T1_beamOFF= np.array([wire.T_background -y_0+ C1(y_0,y_0)*np.cosh(np.lib.scimath.sqrt(A)*x) + C2(y_0,y_0)*np.sinh(np.lib.scimath.sqrt(A)*x) for x in L1])
    T2_beamOFF= np.array([wire.T_background -y_0 + C5(y_0,y_0)*np.cosh(np.lib.scimath.sqrt(A)*x) for x in L2])
    T3_beamOFF = np.array([wire.T_background -y_0 + C1(y_0,y_0)*np.cosh(np.lib.scimath.sqrt(A)*x) - C2(y_0,y_0)*np.sinh(np.lib.scimath.sqrt(A)*x) for x in L3])
    T_beamOFF=np.concatenate((T1_beamOFF,T2_beamOFF,T3_beamOFF))
    R1=R_0*wire.l_wire*(1-wire.a_temperature_coefficient*wire.T_0)
    for T in T_beamOFF:
        R1+=R_0*wire.a_temperature_coefficient*np.real(T)*wire.l_wire/900

    R2= 0.3871*np.real(np.average(T_beamOFF)) - 40.378
    T_av.append(np.max(np.real(T_beamOFF)-273.15))
    F_rad=[]
    F_el=[]
    F_cond=[]
    for T in T_beamOFF:
        F_rad.append(4*np.pi*wire.sigma_stefan_boltzmann*wire.emissivity*wire.d_wire*wire.T_0**3*(np.real(T)-wire.T_0))
        F_el.append(i**2*(R_0+R_0*wire.a_temperature_coefficient*(np.real(T)-wire.T_0)))
    F_rad=np.array(F_rad)
    F_el=np.array(F_el)
    F_cond= F_el-F_rad
    P_rad=0
    for f1 in F_rad:
        P_rad+= f1*wire.l_wire/900
    P_radiation.append(P_rad)
    P_el=0
    for f2 in F_el:
        P_el+= f2*wire.l_wire/900
    P_electrique.append(P_el)
    P_cond=0
    for f4 in F_cond:
        P_cond+= f4*wire.l_wire/900
    P_conduction.append(P_cond)
    P_backgroundgaz.append(0)
P_radiation=np.array(P_radiation)
P_electrique=np.array(P_electrique)
P_conduction=np.array(P_conduction)
P_backgroundgaz=np.array(P_backgroundgaz)


#simulation plots then


dir = "wire_detector/scripts/current_sims_2/0.03mbar_air/"
results_dir = dir + "results/"

P_el_data=power['P_el(mW)']
P_el_data=P_el_data[170:211]
T=power['T (°C)']
T=T[170:211]


T_c=conductivity['Temperature (K)']
T_c=T_c[22:29]
print(T_c)
Cond=conductivity['Thermal Conductivity (W/m-K)']
Cond=Cond[22:29]

cond=interpolate.interp1d(T_c,Cond)
T_new2=np.arange(290,737,1)
cond1=cond(T_new2)

i1=[i*10**-5 for i in range(1,9,3)]
i2=[i*10**-5 for i in range(10,90,10)]
i3=[i*10**-5 for i in range(100,200,5)]
i4 = [i*10**-5 for i in range(200,300,10)]
i_current_list2= i1+i2+i3+i4
P_rad_sim=[]
P_el_sim=[]
P_cond_sim=[]
T_av_sim=[]
P_background_gas_sim=[]
P_cond_sim_T=[]
P_cond_interpolation=[]
for i_current in i_current_list2:
    run_name=  "lw_{}_i_{}".format(2.7,i_current)
    wire = Wire().load(results_dir + run_name)
    func_list = ["f_el", "f_rad", "f_conduction"]
    T_beam_off = np.array(wire.record_dict["T_distribution"][-1])
    T_av_sim.append(np.average(T_beam_off)-273.15)
    F_rad_sim= np.array([wire.f_rad(j) for j in range(wire.n_wire_elements)])
    F_el_sim = np.array([wire.f_el(j) for j in range(wire.n_wire_elements)])
    F_background_gas_sim = np.array([wire.f_background_gas(j) for j in range(wire.n_wire_elements)])
    def k_heat_conductivity(T):
        k_heat_conductivity = -3E-08*T**3 + 0.0001*T**2 - 0.2106*T + 223.82
        return k_heat_conductivity
    def f_conduction1(wire, i):
            T = T_beam_off
            T1=np.average(T_beam_off)
            if i == 0:
                q = (- k_heat_conductivity(T1) * (2 * (wire.T_background - T[i])
                     + (T[i+1] - T[i]))
                     / wire.l_segment)
            elif i == wire.n_wire_elements - 1:
                q = (- k_heat_conductivity(T1) * (2 * (wire.T_background - T[i])
                     + (T[i-1] - T[i]))
                     / wire.l_segment)
            else:
                q = (- k_heat_conductivity(T1) * ((T[i-1] - T[i])
                     + (T[i+1] - T[i])) / wire.l_segment)
            f = q * wire.A_cross_section / wire.l_segment
            return f
    def f_conduction2(wire, i):
            T = T_beam_off
            T1=np.average(T_beam_off)
            if i == 0:
                q = (- cond(T1) * (2 * (wire.T_background - T[i])
                     + (T[i+1] - T[i]))
                     / wire.l_segment)
            elif i == wire.n_wire_elements - 1:
                q = (- cond(T1) * (2 * (wire.T_background - T[i])
                     + (T[i-1] - T[i]))
                     / wire.l_segment)
            else:
                q = (- cond(T1) * ((T[i-1] - T[i])
                     + (T[i+1] - T[i])) / wire.l_segment)
            f = q * wire.A_cross_section / wire.l_segment
            return f
    F_conduction_interpolation=np.array([f_conduction2(wire,j)
                            for j in range(wire.n_wire_elements)])
    F_conduction_sim1 = np.array([f_conduction1(wire,j)
                            for j in range(wire.n_wire_elements)])
    F_conduction_sim = np.array([wire.f_conduction(j)
                            for j in range(wire.n_wire_elements)])
    P_rad1=0
    for f in F_rad_sim:
        P_rad1+= f*wire.l_wire/wire.n_wire_elements
    P_rad_sim.append(P_rad1)
    P_el1=0
    for f in F_el_sim:
        P_el1+= f*wire.l_wire/wire.n_wire_elements
    P_el_sim.append(P_el1)
    P_cond1=0
    for f in F_conduction_sim:
        P_cond1+= f*wire.l_wire/wire.n_wire_elements
    P_cond_sim.append(P_cond1)
    P_cond2=0
    for f in F_conduction_sim1:
        P_cond2+= f*wire.l_wire/wire.n_wire_elements
    P_cond_sim_T.append(P_cond2)
    P_cond3=0
    for f in F_conduction_interpolation:
        P_cond3+= f*wire.l_wire/wire.n_wire_elements
    P_cond_interpolation.append(P_cond3)
    P_bkg=0
    for f in F_background_gas_sim:
        P_bkg+= f*wire.l_wire/wire.n_wire_elements
    P_background_gas_sim.append(P_bkg)

P_cond_bar=[]
P_rad_bar=[]
T_av_bar=[]
i3=[i*10**-5 for i in range(100,200,5)]
i4 = [i*10**-5 for i in range(200,300,10)]
i5 = [i*10**-5 for i in range(300,400,10)]
i_current_list= i3+i4+i5
for i_current in i_current_list:
    run_name=  "lw_P=0.06mbar_{}_i_{}".format(2.7,i_current)
    wire = Wire().load(results_dir + run_name)
    T_beam_off = np.array(wire.record_dict["T_distribution"][-1])
    T_av_bar.append(np.average(T_beam_off)-273.15)
    F_conduction_bar = np.array([wire.f_conduction(j)
                            for j in range(wire.n_wire_elements)])
    F_rad_bar= np.array([wire.f_rad(j) for j in range(wire.n_wire_elements)])
    P_cond_b=0
    for f in F_conduction_bar:
        P_cond_b+= f*wire.l_wire/wire.n_wire_elements
    P_cond_bar.append(P_cond_b)
    P_rad_b=0
    for f in F_rad_bar:
        P_rad_b+= f*wire.l_wire/wire.n_wire_elements
    P_rad_bar.append(P_rad_b)

P_rad_bar=np.array(P_rad_bar)
P_cond_bar=np.array(P_cond_bar)
T_av_bar=np.array(T_av_bar)
T_av_sim=np.array(T_av_sim)
P_rad_sim=np.array(P_rad_sim)
P_el_sim=np.array(P_el_sim)
P_cond_sim=np.array(P_cond_sim)
P_cond_sim_T=np.array(P_cond_sim_T)
P_cond_interpolation=np.array(P_cond_interpolation)
P_background_gas_sim=np.array(P_background_gas_sim)
print(T_av_bar)

f_rad=interpolate.interp1d(T_av_sim,P_rad_sim*10**3)
T_new=np.arange(30,338,1)
T_new_3=np.arange(51,317,1)
f_rad1=f_rad(T_new)
f_el=interpolate.interp1d(T,P_el_data)
f_el1=f_el(T_new)
f_el_bar=f_el(T_new_3)
f_cond=interpolate.interp1d(T_av_sim,P_cond_sim*10**3)
f_cond1=f_cond(T_new)
f_cond_T=interpolate.interp1d(T_av_sim,P_cond_sim_T*10**3)
f_cond_T1=f_cond_T(T_new)
f_cond_interpolation=interpolate.interp1d(T_av_sim,P_cond_interpolation*10**3)
f_cond_interpolation1=f_cond_interpolation(T_new)
f_rad_b=interpolate.interp1d(T_av_bar,P_rad_bar*10**3)
f_rad_bar=f_rad_b(T_new_3)
f_cond_b=interpolate.interp1d(T_av_bar,P_cond_bar*10**3)
f_cond_bar=f_cond_b(T_new_3)

P_missmatch=f_el1-f_rad1-f_cond1
P_missmatch_T=f_el1-f_rad1-f_cond_T1
P_missmatch_interpolation=f_el1-f_rad1-f_cond_interpolation1
P_missmatch_bar=f_el_bar-f_rad_bar-f_cond_bar
fig = plt.figure(0, figsize=(12,6))
ax1= fig.add_subplot(1,2,1)
# ax1.plot(T_av,P_radiation*10**3, '.', label='P_rad')
# ax1.plot(T_av,P_electrique*10**3, '.', label='P_el')
# ax1.plot(T_av,P_conduction*10**3, '.', label=r'P_cond')
ax1.plot(T_av_sim,P_background_gas_sim*10**3, '.', label='P_background_gas_sim')
ax1.plot(T_av_sim,P_el_sim*10**3, '-', label='P_el_sim')
# ax1.plot(T_av_sim,P_cond_sim*10**3, '.', label='P_cond_sim')
# ax1.plot(T_av_sim,P_cond_sim_T*10**3, '.', label='P_cond_sim_T')
# ax1.plot(T_new,P_missmatch, label='P_missmatch', linewidth=1)
# ax1.plot(T_new,P_missmatch_T, label='P_missmatch_T', linewidth=1)
ax1.plot(T,P_el_data, '.', label='P_el_data')
ax1.plot(T_new, f_rad1, linewidth=1,label='P_rad_sim')
ax1.plot(T_new, f_cond1, linewidth=1,label='P_cond_sim')
ax1.plot(T_new, f_cond_T1, linewidth=1,label='P_cond_sim_T')
ax1.set_xlabel('T_average [°C]')
ax1.set_ylabel('Power [mW]')
plt.legend()
plt.grid()
ax2= fig.add_subplot(1,2,2)
ax2.plot(T_av,P_backgroundgaz, '.', label='P_backgroundgaz')
ax2.plot(T_new,P_missmatch, label='P_missmatch', linewidth=1)
ax2.plot(T_new_3,P_missmatch_bar, label='P_missmatch_bar', linewidth=1)
ax2.plot(T_new,P_missmatch_T, label='P_missmatch_T', linewidth=1)
ax2.plot(T_new,P_missmatch_interpolation, label='P_missmatch_interpolation', linewidth=1)
plt.legend()
ax2.set_xlabel('T_average [°C]')
ax2.set_ylabel('Power [mW]')
plt.tight_layout()


plt.show()
