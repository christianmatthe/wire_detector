import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as mpatches
import os
import dill
import pandas as pd
from scipy import interpolate

class Wire():
    """
    "The Wire object holds information about the state of the wire"

    Parameters
    ----------
    n_wire_elements :  `int`
        Number of elements in the wire
    d_wire : `float`
        Diameter of the wire in meters
    l_wire : `float`
        Length of the wire in meters
    k_heat_conductivity : `float` or `path`
        Heat conductivity of the wire matertial in (W/(m*K))
        If `float` k_heat_conductivity will be used for all Temperatures.
        If `str` an interpolation of the Temperature dependence will be 
        generated in k_heat_cond_function. Valid strings can
        be found in "gen_k_heat_cond_function"
    a_temperature_coefficient : `float`
        Temperature coefficient of resistance
    rho_specific_resistance : `float`
        specific resitance fo wire material in (Ohm m)
    i_current : `float`
        current passed through the wire in (A)
    T_background : `float`
        Temperature of "background material" i.e. the holding structure 
        and vacuum chamber
    emissivity : `float`
        emissivity of the wire
    E_recombination : `float`
        Energy released per recombining H_2 molecule in (Joules)
    phi_beam : `float`
        Total atom flux in the beam in (atoms/s)
    l_beam : `float`
        Length of the wire that is illuminated by the beam. equivalent to beam
        width for a centered beam. (meters)
    sigma_beam : `float`
        standart deviation of gaussian in case beam shape is "Gaussian" 
        (meters)
    x_offset_beam : `float`
        offset of the beam center for the wire center along the length of the 
        wire (meters)
    c_specific_heat : `float`
        specific heat capacity of wire material. (J/(kg K))
    density : `float`
        density of wire material. (kg/m**3)
    beam_shape : `str`
        Keyword selection of implemented beam shapes. Currently available are 
        "Gaussian" and "Flat"
    T_base : `array of floats`
        Temperature distribution along wire used at the start of the simulation
        If "None" a uniform distribution at T_background is used. (K)
    taylor_f_rad: `Bool`
        If True f_rad will be calculated with a first order approximation
    crack_eff : `Float`
        Cracking efficiency of Hydrogen cracker. determines percentage of H
        atoms vs H_2 mmolecules in the beam.
    T_cracker: `Float`
        Temperature of the cracker filament. Used to determine blackbody
        radiation from the filament that heats the wire if in line of sight. 
        (K)
    T_atoms : `Float`
        Temperature of the atoms comming out of the cracker. (K)
    T_molecules : `Float`
        Temperature of the atoms comming out of the cracker. (K)
    pressure : `Float`
        Pressure of  Gas surrounding the wire (Pa)
    A_cracker : `Float`
        Area of the cracker opening visible to the wire (m^2)
    dist_cracker_wire: `Float`
        Distance between the cracker opening and the wire (m)
    bodge : `Bool`
        Activates functions I  do not recommend using. Leave on False
    p_laser: `Float`
        Total power in laser beam (Watts)
    m_molecular_gas : `Float`
        mass per molecule of gas surrounding the  wire (kg)
    
    """

    def __init__(self,
                 n_wire_elements = 100,
                 d_wire = 10 * 10**-6,
                 l_wire = 5.45 * 10**-2,
                 k_heat_conductivity = 174,  # Pure Tungsten
                 k_heat_conductivity_interpolateQ = False,
                 a_temperature_coefficient = 4.7 * 10**-3,
                 rho_specific_resistance = 0.052 * 10**-6,
                 i_current = 1.0 * 10**-3,
                 T_background = 293.15,
                 emissivity = 0.2,
                 E_recombination = 7.61 * 10**-19,
                 phi_beam = 1 * (10**17),  # Atoms per second in entire beam
                 l_beam = 25 * 10**-3,
                 sigma_beam = 2 * 10**-3, # from G. Schwendler Thesis
                 x_offset_beam = 0, 
                 c_specific_heat = 133,  # Pure Tungsten
                 density = 19300,  # Pure Tungsten
                 beam_shape = "Gaussian",
                 T_base = None,
                 taylor_f_rad = False,
                 crack_eff = 1,
                 T_cracker = 2300,
                 T_atoms = 2300,
                 T_molecules = 293.15,
                 pressure = 0,
                 A_cracker = np.pi * (0.5 * 10**-3)**2,  # 1mm diameter disk
                 dist_cracker_wire = 100 * 10**-3,
                 bodge = False,
                 p_laser = 0,
                 m_molecular_gas = 2 * 1.674 * 10**-27 
                 # 2 * self.mass_hydrogen
                 ):

        self.n_wire_elements = n_wire_elements
        self.d_wire = d_wire
        self.l_wire = l_wire
        self.k_heat_conductivity = k_heat_conductivity
        self.a_temperature_coefficient = a_temperature_coefficient
        self.rho_specific_resistance = rho_specific_resistance
        self.i_current = i_current
        self.T_background = T_background
        self.emissivity = emissivity
        self.E_recombination = E_recombination
        self.phi_beam = phi_beam
        self.l_beam = l_beam
        self.sigma_beam = sigma_beam
        self.x_offset_beam = x_offset_beam
        self.c_specific_heat = c_specific_heat
        self.density = density
        self.T_base = T_base
        self.beam_shape = beam_shape
        self.taylor_f_rad = taylor_f_rad
        self.crack_eff = crack_eff
        self.T_cracker = T_cracker
        self.T_atoms = T_atoms
        self.T_molecules = T_molecules
        self.pressure = pressure
        self.A_cracker = A_cracker
        self.dist_cracker_wire = dist_cracker_wire
        self.bodge = bodge
        self.p_laser = p_laser
        self.m_molecular_gas = m_molecular_gas

        #Constants
        self.T_0 = 293.15
        self.sigma_stefan_boltzmann = 5.6704 * 10**-8
        self.k_boltzmann = 1.38064852 * 10**-23
        self.mass_hydrogen = 1.674 * 10**-27

        #derived paramters
        self.A_cross_section = (np.pi /4) * self.d_wire ** 2
        self.A_surface = np.pi * self.d_wire * self.l_wire 
        self.l_segment = self.l_wire / self.n_wire_elements

        #Intial State of the Wire
        self.T_distribution = np.asarray(
            [self.T_background for n in range(self.n_wire_elements)] )
        self.simulation_time = 0
        self.gen_k_heat_cond_function()

    def resistance_segment(self, i, T_dist=None):
        if T_dist is None:
            T = self.T_distribution[i]
        else:
            T = T_dist[i]
        r = ((self.rho_specific_resistance * self.l_segment)
             /self.A_cross_section) * (1 + self.a_temperature_coefficient
             * (T - self.T_0))
        return r
    
    def resistance_total(self, T_dist=None):
        if T_dist is None:
            T = self.T_distribution
        else:
            T = T_dist
        R = np.sum([self.resistance_segment(i, T) 
               for i in range(self.n_wire_elements)])
        return R

    def f_rad(self, i):
        T = self.T_distribution[i]
        if self.emissivity == 0:
            f = 0
        elif self.taylor_f_rad == True:
            f = (np.pi * self.d_wire * self.sigma_stefan_boltzmann
                 * self.emissivity * 4 * self.T_background**3
                 * (T - self.T_background))
        else:
            f = (np.pi * self.d_wire * self.sigma_stefan_boltzmann
                * self.emissivity * (T**4 - self.T_background**4))  
        return f
    
    def f_el(self, i):
        f = (self.i_current**2 * self.resistance_segment(i) / self.l_segment)
        return f

    def gen_k_heat_cond_function(self):
        # Temperature depenent heat conductivity coefficient of Tungsten 
        # based on 
        # https://www.efunda.com/materials/elements/TC_Table.cfm?Element_ID=W
        if (type(self.k_heat_conductivity) == float 
            or type(self.k_heat_conductivity) == int):
            self.k_heat_cond_function = lambda T : self.k_heat_conductivity
        elif type(self.k_heat_conductivity) is str:
            if self.k_heat_conductivity is "interpolate_tungsten":
                # top_dir = os.path.dirname(os.path.abspath(__file__)) + os.sep
                # resource_dir = top_dir + "resources" + os.sep
                # file_path = (resource_dir 
                #             + 'Thermal_conductivity_of_Tungsten.csv')
                # data = pd.read_csv(file_path)
                # T_list = data['Temperature (K)'].values.tolist()
                # k_list = data['Thermal Conductivity (W/m-K)'].values.tolist()
                T_list = [1, 2, 3,4,5,6,7,8,9,10,15,20,30,40,50,60,70,80,90,
                          100, 150,200,250,300, 350, 400, 500, 600, 800, 1000,
                          1200,1400, 1600, 1800, 2000]
                k_list = [1440, 2870, 4280, 5630, 6870, 7950, 8800, 9380, 9680,
                          9710, 7200, 4050, 1440, 692, 427, 314, 258, 229, 217,
                          208, 192, 185, 180, 174, 167, 159, 146, 137, 125, 
                          118, 112, 108, 104, 101, 98]

                self.k_hc_interpolation = interpolate.interp1d(T_list, k_list,
                                                          "cubic")
                self.k_heat_cond_function = self.k_hc_interpolation
            else:
                raise Exception('"k_heat_conductivity" is not a valid `str`')
        else: 
            raise Exception('"k_heat_conductivity" is not a valid type')
    
    
    def f_conduction(self, i):
        # Linear interpolation approximation to differential across segment
        T = self.T_distribution
        #q: Total Heat flow per square meter out of Wire segment
        #NOTE: must be impplemented as -f_conduction -> Heat is gained if
        #      colder than surroundings and lost if Hotter

        #Boundary Conditions:
        if i == 0:
            q = (- self.k_heat_cond_function(T[i]) 
                 * (2 * (self.T_background - T[i]) + (T[i+1] - T[i]))
                 / self.l_segment)
        elif i == self.n_wire_elements - 1:
            q = (- self.k_heat_cond_function(T[i]) 
                 * (2 * (self.T_background - T[i]) + (T[i-1] - T[i]))
                 / self.l_segment)       
        else:
            q = (- self.k_heat_cond_function(T[i]) 
                 * ((T[i-1] - T[i]) + (T[i+1] - T[i])) 
                 / self.l_segment)
        #multiply with area and divide by l_segment to get watts per m
        # wire length to get f
        f = q * self.A_cross_section / self.l_segment 
        return f

    def f_beam(self, i):
        if self.beam_shape == "Flat":
            # flat circular beam profile
            x_pos = ((i + 0.5) * self.l_segment - (self.l_wire / 2))
            if ((-self.l_beam / 2) + self.x_offset_beam < x_pos 
                and x_pos < (self.l_beam / 2) + self.x_offset_beam): 
                f = ((2 * self.phi_beam * self.d_wire * self.E_recombination) 
                 /(np.pi * self.l_beam**2))
            else:
                f = 0
        elif self.beam_shape == "Gaussian":
            # Gaussian Profile
            x_pos = ((i + 0.5) * self.l_segment - (self.l_wire / 2))
            y_pos = 0
            f = (2 * self.d_wire * self.E_recombination
                 * self.phi_beam * (1/(2 * np.pi * self.sigma_beam ** 2)) 
                 * np.exp((-1/2) * ((x_pos - self.x_offset_beam)
                 / self.sigma_beam) 
                 ** 2 + ((y_pos)/self.sigma_beam)) ** 2) 
        else:
            raise Exception("Unrecognized beam shape")
        return f

    def f_laser(self, i):
        if self.beam_shape == "Flat":
            # flat circular profile
            x_pos = ((i + 0.5) * self.l_segment - (self.l_wire / 2))
            if ((-self.l_beam / 2) + self.x_offset_beam < x_pos 
                and x_pos < (self.l_beam / 2) + self.x_offset_beam): 
                f = (( self.emissivity * self.p_laser * self.d_wire) 
                 /(np.pi * (self.l_beam/2)**2))
            else:
                f = 0
        elif self.beam_shape == "Gaussian":
            # Gaussian Profile
            x_pos = ((i + 0.5) * self.l_segment - (self.l_wire / 2))
            y_pos = 0
            f = (( self.emissivity * self.p_laser * self.d_wire)
                 * (1/(2 * np.pi * self.sigma_beam ** 2)) 
                 * np.exp((-1/2) * ((x_pos - self.x_offset_beam)
                 / self.sigma_beam) ** 2 + ((y_pos)/self.sigma_beam)) ** 2) 
        else:
            raise Exception("Unrecognized beam shape")
        return f

    def f_bb(self, i):
        # Blackbody radiation from cracker. diameter 1mm stefan boltzmann 
        # times wire area and emissivity as absorption coefficient
        # (assume flat absorbtion profile)

        # flat circular beam profile
        x_pos = ((i + 0.5) * self.l_segment - (self.l_wire / 2))
        A_sphere = 4 * np.pi * self.dist_cracker_wire**2
        A_incident = self.l_beam * self.d_wire
        if ((-self.l_beam / 2) + self.x_offset_beam < x_pos
            and x_pos < (self.l_beam / 2) + self.x_offset_beam): 
            f = (self.emissivity # Note: Emissivity = absorbtivity of wire
                 * self.sigma_stefan_boltzmann * self.T_cracker**4
                 * self.A_cracker * (A_incident/A_sphere) 
                 / self.l_beam) # normalization to length density
        else:
            f = 0
        return f



    def f_beam_gas(self, i):
        n_atoms = self.f_beam(i) / (self.E_recombination/2)
        n_molecules = (n_atoms/2) * (1/self.crack_eff - 1)
        f_atoms = (n_atoms * (3/2) * self.k_boltzmann 
                   * (self.T_atoms - self.T_distribution[i]))
        f_molecules = (n_molecules * (3/2) * self.k_boltzmann 
                   * (self.T_molecules - self.T_distribution[i]))
        f = f_atoms + f_molecules
        return f
    
    def f_background_gas(self, i):
        # (average) mass of background Gas. Assumes background gas is primarily 
        # due to beam gas and all components have the same pumping speeds
        #m = 2 * self.mass_hydrogen
        m = self.m_molecular_gas
        # m = (self.mass_hydrogen * self.crack_eff 
        #     + 2 * self.mass_hydrogen * (1 - self.crack_eff))
        # calculate power density
        q = ((self.pressure/4) * np.sqrt(3 * self.k_boltzmann / m) 
           * (self.T_distribution[i] - self.T_background)
           /np.sqrt(self.T_background))
        # multiply with total outward surface area of wire segment 
        f = q * np.pi * self.d_wire *self.l_segment / self.l_segment

        #print("f:", f)
        #print("Tdist[{}]:".format(i), self.T_distribution[i])
        if np.isnan(q):
            print("f:", f)
            print("Tdist[i]:", self.T_distribution[i])
            raise ValueError('q_background_gas is NAN')
        return f

    def f_conduction_bodge(self, i):
        # Linear interpolation approximation to differential across segment
        T = self.T_distribution
        #q: Total Heat flow per square meter out of Wire segment
        #NOTE: must be impplemented as -f_conduction -> Heat is gained if
        #      colder than surroundings and lost if Hotter
        factor = 100
        #Boundary Conditions:
        if i == 0:
            q = (- self.k_heat_cond_function(T[i]) 
                 * (2 * (self.T_background - T[i]) 
                 + (T[i+1] - T[i])) * factor
                 / self.l_segment)
        elif i == self.n_wire_elements - 1:
            q = (- self.k_heat_cond_function(T[i]) 
                 * (2 *(self.T_background - T[i]) 
                 + (T[i-1] - T[i])) * factor
                 / self.l_segment)       
        else:
            # Increase heat conductivity in ends by 100x for illustration
            if (i < 25 or i >self.n_wire_elements - 1 - 25):
                q = (- factor * self.k_heat_cond_function(T[i]) 
                     * ((T[i-1] - T[i]) 
                     + (T[i+1] - T[i])) / self.l_segment)
            elif i == 25:
                q = (- self.k_heat_cond_function(T[i]) 
                    * (factor *(T[i-1] - T[i]) 
                    + (T[i+1] - T[i])) / self.l_segment)
            elif i == self.n_wire_elements - 1 - 25:
                q = (- self.k_heat_cond_function(T[i]) 
                    * ((T[i-1] - T[i]) 
                    + factor * (T[i+1] - T[i])) / self.l_segment)
            else:
                q = (- self.k_heat_cond_function(T[i]) * ((T[i-1] - T[i]) 
                    + (T[i+1] - T[i])) / self.l_segment)
        #multiply with area and divide by l_segment to get watts per m
        # wire length to get f
        f = q * self.A_cross_section / self.l_segment 
        return f

    def temperature_change(self, i, time_step):
        delta_T = (((self.f_el(i) - self.f_rad(i) - self.f_conduction(i)
                     + self.f_beam(i) + self.f_beam_gas(i) + self.f_bb(i)
                     - self.f_background_gas(i) + self.f_laser(i)
                     ) 
                     * self.l_segment * time_step) 
                     / (self.density * self.A_cross_section * self.l_segment 
                     * self.c_specific_heat))
        # deprecated             
        if self.bodge == True:
            if (i < (25+1) or i >self.n_wire_elements - 1 - (25+1)):
                delta_T = (((0*self.f_el(i) - self.f_rad(i) 
                        - self.f_conduction_bodge(i)
                        + self.f_beam(i) + self.f_beam_gas(i) + self.f_bb(i)) 
                        * self.l_segment * time_step) 
                        / (self.density * self.A_cross_section * self.l_segment 
                        * 100* self.c_specific_heat))
            else:
                delta_T = (((self.f_el(i) - self.f_rad(i) 
                        - self.f_conduction_bodge(i)
                        + self.f_beam(i) + self.f_beam_gas(i) + self.f_bb(i)) 
                        * self.l_segment * time_step) 
                        / (self.density * self.A_cross_section * self.l_segment 
                        * self.c_specific_heat))
        return delta_T

    def simulation_step(self, time_step = 0.0001):
        T_dist_new = np.asarray([self.T_distribution[i] 
                        + self.temperature_change(i, time_step)
                        for i in range(self.n_wire_elements)])
        self.T_distribution = T_dist_new
        self.simulation_time = self.simulation_time + time_step
        return None

    # TODO Implement Check if previous simulation exists and load from file
    # TODO Dynamic step syste based on ratio of heat flow and heat capacity
    def simulate(self, n_steps = 15000, record_steps = 150,
                 time_step = 0.001):
        self.record_dict = {}
        self.record_dict["T_distribution"] = np.zeros(
            (record_steps + 1, self.n_wire_elements))
        self.record_dict["time"] = np.zeros(record_steps + 1)

        #Intial State of the Wire
        self.simulation_time = 0
        if self.T_base is None:
            self.T_distribution = (np.asarray([self.T_background 
                                for n in range(self.n_wire_elements)]))
        else:
            self.T_distribution = self.T_base
        

        for i in range(n_steps + 1):
            if i in range(0, n_steps + 1, (n_steps // record_steps)):  
                self.record_dict["T_distribution"][
                    i // ((n_steps // record_steps))] = (
                    self.T_distribution)
                self.record_dict["time"][
                    i // ((n_steps // record_steps))] = (
                    self.simulation_time)
            self.simulation_step(time_step = time_step)
        return None

    # Plan:
    # write functions to reconstruct wire properties from pickeld record
    # without resimulation
    # (Interpolate between recorded steps?)
    # resistance, signal, x_lst
    # 
    # Write default plot and animation functions

    def save(self, filename):
        with open(filename + ".pkl", "wb") as f:
            dill.dump(self, f)

    def load(self, filename):
        with open(filename + ".pkl", "rb") as f:
            wire = dill.load(f)
        return wire

    def U_wire(self, i):
        # Calculate volatage drop over wire
        U = (self.resistance_total(self.record_dict["T_distribution"][i])
                   * self.i_current)
        return U

    def integrate_f(self, f_func):
        lst = [f_func(j) for j in range(self.n_wire_elements)]
        arr = np.array(lst)
        # Calculate total power along wire
        power = np.sum(arr * self.l_segment)
        return power


    def plot_signal(self, filename="plots/signal_plot"):
        # Plot Temperature over Wire for start and end of simulation
        # Plot Temperature over Wire
        plt.figure(0, figsize=(8,6.5))
        ax1 = plt.gca()

        x_lst = [1000 * ((i + 0.5) * self.l_segment - (self.l_wire / 2))
                for i in range(self.n_wire_elements)]
        T_beam_off = self.record_dict["T_distribution"][0]
        T_beam_on = self.record_dict["T_distribution"][-1]
        T_lst = [T_beam_off, T_beam_on]

        R_arr = np.zeros(2)
        for i,T_dist in enumerate(T_lst):
            self.T_distribution = T_dist
            R_arr[i] = self.resistance_total()
        
        U_delta = (R_arr[1] - R_arr[0]) * self.i_current
        signal = (R_arr[1] - R_arr[0])/R_arr[0]

        ax1.plot(x_lst, T_lst[0] - 273.15, "-", label=r"Beam Off, " 
                 + "R = {:.3f}".format(R_arr[0]) + r"$\Omega$")
        ax1.plot(x_lst, T_lst[1] - 273.15, "-", label=r"Beam On, " 
                 + "R = {:.3f}".format(R_arr[1]) + r"$\Omega$")
                 
        ax1.set_ylabel("Temperature [??C]")
        ax1.set_xlabel(r"wire positon [mm]")
        plt.title(r"$d_{wire}$ = " + "{}".format(self.d_wire * 10**6) 
                  + r"$\mu m$" +", I = " + "{}".format(self.i_current * 10**3)
                  + r"$mA$" + r", $\phi_{beam}$ = 10^" + "{:.2f}".format(
                  np.log10(self.phi_beam)))
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
        plt.savefig(filename + '.{}'.format(format_im),
                    format=format_im, dpi=dpi)
        ax1.cla()

    def plot_R_over_t(self, filename="plots/R_over_t"):
        # Plot Resistance over time
        plt.figure(0, figsize=(8,6.5))
        ax1=plt.gca()

        t_lst = self.record_dict["time"]
        steps = len(t_lst)
        R_lst = [self.resistance_total(self.record_dict["T_distribution"][i])
                 for i in range(steps)]

        R_tau = R_lst[0] + (R_lst[-1] - R_lst[0])*(1 - 1/np.exp(1))
        R_tau_lst = [R_tau for i in range(len(t_lst))]
        R_95 = R_lst[0] + (R_lst[-1] - R_lst[0])*0.95
        R_95_lst = [R_95 for i in range(len(t_lst))]
        # calculate time at which these are reached
        t_tau = t_lst[np.argmin(np.absolute(R_lst - R_tau))]
        t_95 = t_lst[np.argmin(np.absolute(R_lst - R_95))]
        
        ax1.plot(t_lst, R_lst, "-", label="Resistance")
        ax1.plot(t_lst, R_tau_lst, "-",
                 label=r"$\Delta R \cdot$(1 - 1/e)" + ", t = {:.3f}".format(t_tau))
        ax1.plot(t_lst, R_95_lst, "-",
                 label=r"$0.95 \cdot \Delta R$" + ", t = {:.3f}".format(t_95))
        ax1.set_ylabel(r"Resistance [$\Omega$]")
        ax1.set_xlabel(r"time [s]")
        plt.grid(True)
        plt.legend(shadow=True)

        format_im = 'png' #'pdf' or png
        dpi = 300
        plt.savefig(filename + '.{}'.format(format_im),
                    format=format_im, dpi=dpi)
        ax1.cla()

    def plot_heat_flow(self, filename="plots/heat_flow", log_y = False):
        # Calculate endstate of heat flow
        x_lst = [1000 * ((i + 0.5) * self.l_segment - (self.l_wire / 2))
                for i in range(self.n_wire_elements)]
        self.T_distribution = self.record_dict["T_distribution"][-1]
        f_el_arr = [self.f_el(j) for j in range(self.n_wire_elements)]
        f_conduction_arr = [self.f_conduction(j) 
                            for j in range(self.n_wire_elements)]
        f_rad_arr = [self.f_rad(j) for j in range(self.n_wire_elements)]
        f_beam_arr = [self.f_beam(j) for j in range(self.n_wire_elements)]
        f_beam_gas_arr = [self.f_beam_gas(j) 
                          for j in range(self.n_wire_elements)]
        f_bb_arr = [self.f_bb(j) for j in range(self.n_wire_elements)]
        f_background_gas_arr = [self.f_background_gas(j)
                                for j in range(self.n_wire_elements)]
        f_laser_arr = [self.f_laser(j)
                                for j in range(self.n_wire_elements)]

        f_conduction_bodge_arr = [self.f_conduction_bodge(j) 
                        for j in range(self.n_wire_elements)]

        # Plot endstate of heat flow
        plt.figure(0, figsize=(8,6.5))
        ax1=plt.gca()

        # if self.bodge == True:
        #     for i in range(0,25):
        #         f_el_arr[i] = 0
        #     for i in range(self.n_wire_elements - 25,self.n_wire_elements):
        #         f_el_arr[i] = 0
        ax1.plot(x_lst, f_el_arr, "-", label=r"$F_{el}$")
        #bodge_start
        try:
            if self.bodge == True:
                ax1.plot(x_lst, f_conduction_bodge_arr, "--"
                        , label=r"$F_{\mathrm{cond. piecewise}}$")
            else:
                ax1.plot(x_lst, f_conduction_arr, "--",
                         label=r"$-F_{conduction}$")
        except:
            #bodge_end
            ax1.plot(x_lst, f_conduction_arr, "--", label=r"$-F_{conduction}$")
        ax1.plot(x_lst, f_rad_arr, "--", label=r"$-F_{rad}$")
        ax1.plot(x_lst, f_beam_arr, "-", label=r"$F_{beam}$")
        ax1.plot(x_lst, f_beam_gas_arr, "-", label=r"$F_{beam \,gas}$")
        ax1.plot(x_lst, f_bb_arr, "-", label=r"$F_{bb\, cracker}$")
        ax1.plot(x_lst, f_background_gas_arr, "--"
                 , label=r"$-F_{backgr. \, gas}$")
        ax1.plot(x_lst, f_laser_arr, "-"
                 , label=r"$F_{laser}$")

        ax1.set_ylabel("Heat Flow [W/m]", fontsize = 16)
        ax1.set_xlabel(r"Wire positon [mm]", fontsize = 16)
        ax1.tick_params(axis='both', which='major', labelsize=12)
        plt.grid(True)
        plt.legend(shadow=True)

        #fancy legend:
        if True:
            h, l = ax1.get_legend_handles_labels()
            sources = [0,3,
                        4,5,7
                        ]
            sinks = [1,2
                     ,6
                     ]
            #TOP legend
            # l1 = ax1.legend([h[i] for i in sources], [l[i] for i in sources],
            #    #shadow = True,
            #    #framealpha = 0.5,
            #    loc = "lower left",
            #    bbox_to_anchor=(0, 1),
            #    fontsize = "large",
            #    title = "Heat Sources:",
            #    ncol = 2
            #    )
            # plt.gca().add_artist(l1)
            # l2 = ax1.legend([h[i] for i in sinks], [l[i] for i in sinks],
            #    #shadow = True,
            #    #framealpha = 0.5,
            #    loc = "lower right",
            #    bbox_to_anchor=(1, 1),
            #    fontsize = "large",
            #    title = "Heat Sinks:",
            #    ncol = 2
            #    )
            #Right Side Legend
            l1 = ax1.legend([h[i] for i in sources], [l[i] for i in sources],
               #shadow = True,
               #framealpha = 0.5,
               loc = "upper left",
               bbox_to_anchor=(1, 1),
               fontsize = 14,
               title = "Heat Sources:",
               title_fontsize = 14,
               ncol = 1
               )
            plt.gca().add_artist(l1)
            ax1.legend([h[i] for i in sinks], [l[i] for i in sinks],
               #shadow = True,
               #framealpha = 0.5,
               loc = "upper left",
               bbox_to_anchor=(1, 0.60),
               fontsize = 14,
               title = "Heat Sinks:",
               title_fontsize = 14,
               ncol = 1
               )
            plt.tight_layout()

        if log_y == True:
            ax1.set_yscale("log")
            if filename == "plots/heat_flow":
                filename= "plots/log_heat_flow"

        format_im = 'png' #'pdf' or png
        dpi = 300
        plt.savefig(filename + '.{}'.format(format_im),
                    format=format_im, dpi=dpi)
        ax1.cla()

            
def pressure_chamber(phi_beam, crack_eff):
    #calculate flow in sccm
    sccm = 4.478 * 10**17
    mbar_to_Pa = 100
    flow_sccm = phi_beam * (1/crack_eff) / sccm
    #Empirical Formula based on elog of Nov 23 2148 10 sccm
    p = (flow_sccm * 1.4 *10**-8 + 5 * 10**-10) * mbar_to_Pa
    return p 


if __name__ == "__main__":
    # Alexandre 96 Paper values
    # wire = Wire(n_wire_elements = 100, k_heat_conductivity = 170,
    #             i_current = 0.1 * 10**-3, d_wire = 10 * 10**-6,
    #             emissivity = 0.3, l_wire=5*10**-2, 
    #             rho_specific_resistance = 668 * (np.pi*((10/2)*10**-6)**2),
    #             beam_shape="Flat"
    # )

    # # AG Pohl values (Beam is not accurate)
    # i_current = 1 * 10**-3
    # d_wire = 10 * 10**-6
    # #taylor_f_rad_list = [False, True]
    # wire = Wire(n_wire_elements = 100, k_heat_conductivity = 174,
    #                     i_current = i_current, d_wire = d_wire,
    #                     emissivity = 0.2, l_wire=5.45*10**-2,
    #                     beam_shape="Gaussian", sigma_beam=6*10**-3, 
    #                     phi_beam=10**17
    #                     #T_base=wire_no_beam.record_dict["T_distribution"][-1]
    #                     ###
    #                     #,taylor_f_rad=taylor_f_rad
    #         ) 
    # n_steps = 15000
    # record_steps = 200
    # time_step = 0.001
    # wire.simulate(n_steps=n_steps, record_steps=record_steps,
    #             time_step=time_step)



    ########################### AG Pohl values (Beam is not accurate)
    i_current = 1 * 10**-3
    d_wire = 20 * 10**-6
    taylor_f_rad_list = [False, True]
    for taylor_f_rad in taylor_f_rad_list:
        wire_no_beam = Wire(n_wire_elements = 100, k_heat_conductivity = 174,
                    i_current = i_current, d_wire = d_wire,
                    emissivity = 0.2, l_wire=5.45*10**-2,
                    ###
                    phi_beam=0, T_base=None
                    ###
                    ,taylor_f_rad=taylor_f_rad
        ) 
        # Run the Simulation
        n_steps = 30000
        record_steps = 150
        time_step = 0.001
        wire_no_beam.simulate(n_steps=n_steps, record_steps=record_steps,
                    time_step=time_step)

        if taylor_f_rad == True:
            taylor_string = "_taylor"
        else:
            taylor_string = ""
        top_dir = "{0:.0f}um_phi_beam_sweep{1}/".format(d_wire*10**6,
                                                        taylor_string)
        os.makedirs(top_dir, exist_ok=True)
        os.makedirs(top_dir + "plots/", exist_ok=True)
        exp_list = np.linspace(15,20,num = 21)
        for phi_exp in exp_list:
            wire = Wire(n_wire_elements = 100, k_heat_conductivity = 174,
                        i_current = i_current, d_wire = d_wire,
                        emissivity = 0.2, l_wire=5.45*10**-2,
                        beam_shape="Gaussian", sigma_beam=6*10**-3, 
                        phi_beam=10**phi_exp,
                        T_base=wire_no_beam.record_dict["T_distribution"][-1]
                        ###
                        ,taylor_f_rad=taylor_f_rad
            ) 
            wire.simulate(n_steps=n_steps, record_steps=record_steps,
                        time_step=time_step)
            wire.plot_signal(top_dir + "plots/phi_to_{}".format(phi_exp))
            wire.save(top_dir + "phi_to_{}".format(phi_exp))


    # Calculate endstate of heat flow
    x_lst = [1000 * ((i + 0.5) * wire.l_segment - (wire.l_wire / 2))
             for i in range(wire.n_wire_elements)]
    wire.T_distribution = wire.record_dict["T_distribution"][-1]
    f_el_arr = [wire.f_el(j) for j in range(wire.n_wire_elements)]
    f_conduction_arr = [wire.f_conduction(j) 
                        for j in range(wire.n_wire_elements)]
    f_rad_arr = [wire.f_rad(j) for j in range(wire.n_wire_elements)]
    f_beam_arr = [wire.f_beam(j) for j in range(wire.n_wire_elements)]

    # Plot endstate of heat flow
    fig = plt.figure(0, figsize=(8,6.5))
    ax1=plt.gca()

    ax1.plot(x_lst, f_el_arr, "-", label=r"$F_{el}$")
    ax1.plot(x_lst, f_conduction_arr, "-", label=r"$F_{conduction}$")
    ax1.plot(x_lst, f_rad_arr, "-", label=r"$F_{rad}$")
    ax1.plot(x_lst, f_beam_arr, "-", label=r"$F_{beam}$")

    ax1.set_ylabel("Heat Flow [W/m]")
    ax1.set_xlabel(r"Wire positon [mm]")
    plt.grid(True)
    plt.legend(shadow=True)
    
    
    format_im = 'png' #'pdf' or png
    dpi = 300
    plt.savefig("plots/Heat_flow_plot" + '.{}'.format(format_im),
                format=format_im, dpi=dpi)
    ax1.cla()


    
    
    