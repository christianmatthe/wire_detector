import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import interpolate
import os
import dill

#NOTE the interpackage dependency
from wire_analysis.beamshape import calc_norm_factor, j


class Wire:
    """
    The Wire object holds information about the state of the wire.

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
        Temperature coefficient of resistance (1/K)
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

    def __init__(
        self,
        n_wire_elements=100,
        d_wire=10 * 10**-6,
        l_wire=5.45 * 10**-2,
        k_heat_conductivity=174,  # Pure Tungsten
        a_temperature_coefficient=4.7 * 10**-3,
        rho_specific_resistance=0.052 * 10**-6,
        i_current=1.0 * 10**-3,
        T_background=293.15,
        emissivity=0.2,
        E_recombination=7.1511 * 10**-19,  # Joules
        # https://journals.aps.org/prl/pdf/10.1103/PhysRevLett.121.013001
        # equivalent to 4.4634 eV
        phi_beam=1 * (10**17),  # Atoms per second in entire beam
        l_beam=25 * 10**-3,
        sigma_beam=2 * 10**-3,  # from G. Schwendler Thesis
        x_offset_beam=0,
        c_specific_heat=133,  # Pure Tungsten
        density=19300,  # Pure Tungsten
        beam_shape="Gaussian",
        T_base=None,
        taylor_f_rad=False,
        crack_eff=1,
        T_cracker=2300,
        T_atoms=2300,
        T_molecules=293.15,
        pressure=0,
        A_cracker=np.pi * (0.5 * 10**-3) ** 2,  # 1mm diameter disk
        dist_cracker_wire=100 * 10**-3,
        bodge=False,
        p_laser=0,
        m_molecular_gas=2 * 1.674 * 10**-27,
        # for j Tschersich:
        y0 = 35.17 * 10**-3, # m, wire to HABS distance in CAD)
        l_eff = 4.0,
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

        # Constants
        self.T_0 = 293.15
        self.sigma_stefan_boltzmann = 5.6704 * 10**-8
        self.k_boltzmann = 1.38064852 * 10**-23
        self.mass_hydrogen = 1.674 * 10**-27

        # derived paramters
        self.A_cross_section = (np.pi / 4) * self.d_wire**2
        self.A_surface = np.pi * self.d_wire * self.l_wire
        self.l_segment = self.l_wire / self.n_wire_elements
        self.gen_k_heat_cond_function()

        # Intial State of the Wire
        self.T_distribution = np.ones(self.n_wire_elements) * self.T_background
        self.simulation_time = 0

        # Initialize tschersich beam vars
        self.y0  = y0 # m, wire to HABS distance in CAD)
        self.l_eff = l_eff
        #init norm_factor
        # "beamshape.py" speaks milimeters rather than meters
        # calc_norm_dactor,, does not even actually use y0, and does not need
        # to ooof.
        # But python throws an error
        # NOTE to self, stick to SI base units
        # adjust accordingly
        self.norm_factor  = calc_norm_factor(l_eff = self.l_eff,
                                        #y0 = self.y0 * 1000
                                        )
        return
    
    
    # if self.beam_shape == "Tschersich":
    #NOTE the interpackage dependency
    def theta(self,x):
        y0 = self.y0
        z0 = 0 # simple casse of centetered wire
        z_center = 0 # simple casse of centetered wire
        return np.arctan(np.sqrt(x ** 2 + (z_center - z0) ** 2) / y0)

    def j_norm_linear(self,
                x,
                ) -> np.ndarray:
        """
        This function serves to integrate the H distribution along a thin rectangle
        i.e. a projected wire. 
        DO NOT ENTER LARGE z_lims. Keep them well below mm

        The simplification to a 1D integral greatly speeds up the integration
        """
    
        j_norm_lin =  ( self.norm_factor
                            # * z_width # will move to f_beam
                            * j(self.theta(x), self.l_eff)
                            * 1/(self.y0**2 * (1/np.cos(self.theta(x))**3))
                            # from solid angle to area on plane
                                )
        return j_norm_lin



    def gen_k_heat_cond_function(self) -> None:
        """
        Generate conductivity function. If `self.k_heat_conductivity` is a number,
        this constant is used. If `self.k_heat_conductivity` is "interpolate_tungsten",
        a temperature dependent heat conductivity is used based on:
        https://www.efunda.com/materials/elements/TC_Table.cfm?Element_ID=W
        The resulting function is stored in `self.k_heat_cond_function`.
        """
        if (
            type(self.k_heat_conductivity) == float
            or type(self.k_heat_conductivity) == int
        ):
            self.k_heat_cond_function = lambda T: self.k_heat_conductivity
        elif type(self.k_heat_conductivity) is str:
            if self.k_heat_conductivity == "interpolate_tungsten":
                T_list = [
                    1,2,3,4,5,6,7,8,9,10,15,20,30,40,50,60,70,80,90,100,150,200,
                    250,300,350,400,500,600,800,1000,1200,1400,1600,1800,2000
                ]
                k_list = [
                    1440,2870,4280,5630,6870,7950,8800,9380,9680,9710,
                    7200,4050,1440,692,427,314,258,229,217,208,192,185,
                    180,174,167,159,146,137,125,118,112,108,104,101,98
                ]

                self.k_hc_interpolation = interpolate.interp1d(T_list, k_list, "cubic")
                self.k_heat_cond_function = self.k_hc_interpolation
            else:
                raise Exception('"k_heat_conductivity" is not a valid `str`')
        else:
            raise Exception('"k_heat_conductivity" is not a valid type')

    def resistance_segment(self, T_dist: np.ndarray) -> np.ndarray:
        """Return resistance of the wire segments as an array."""
        return (
            (self.rho_specific_resistance * self.l_segment) / self.A_cross_section
        ) * (1 + self.a_temperature_coefficient * (T_dist - self.T_0))

    def resistance_total(self, T_dist: np.ndarray = None) -> float:
        """Compute resistance of full wire (= sum over all segments)."""
        if T_dist is None:
            T = self.T_distribution
        else:
            T = T_dist
        R = np.sum(self.resistance_segment(T))
        return R

    def f_rad(self) -> np.ndarray:
        """Heat transfer per unit length due to blackbody radiation/absorption of wire"""
        return (
            np.pi
            * self.d_wire
            * self.sigma_stefan_boltzmann
            * self.emissivity
            * (self.T_distribution**4 - self.T_background**4)
        )

    def f_el(self) -> np.ndarray:
        """Heat transfer per unit length due to electrical current"""
        return (
            self.i_current**2
            * self.resistance_segment(self.T_distribution)
            / self.l_segment
        )

    def f_conduction(self) -> np.ndarray:
        """
        Heat transfer due to conduction between wire elements. The first and last
        wire element are connected to an ideal heat sink with temperature
        `self.T_background`. It must be impplemented as `-f_conduction()` because
        heat is gained if colder than surroundings and lost if hotter.
        """
        shift_left = np.roll(self.T_distribution, 1)
        shift_right = np.roll(self.T_distribution, -1)
        shift_left[0] = self.T_background
        shift_right[-1] = self.T_background

        # Difference of ith wire element with (i-1)th or (i+1)th element
        diff_left = shift_left - self.T_distribution
        diff_right = shift_right - self.T_distribution
        # Multiply the left and right element by two, because the distance
        # to the heat sink is only half the distance between wire elements
        diff_left[0] *= 2
        diff_right[-1] *= 2

        # Numeric approximation of second derivative
        # q: Total Heat flow per square meter out of Wire segment
        q = -self.k_heat_cond_function(self.T_distribution) * (
            (diff_left + diff_right) / self.l_segment
        )
        # multiply with area and divide by l_segment to get watts per m
        # wire length to get heat flow
        return q * self.A_cross_section / self.l_segment

    def f_beam(self) -> np.ndarray:
        """
        Heat Conduction due the recombination of hydrogen atoms.
        Three different beam shapes are available:
        - "Flat": The distribution of atoms is uniformly distributed over a disk with
            radius self.l_beam/2
        - "Gaussian": Gaussian distribution with standard deviation self.sigma_beam
        - "Point": Only a single element is heated by the beam.
        """
        shape = len(self.T_distribution)
        x_positions = (np.arange(shape) + 0.5) * self.l_segment - (self.l_wire / 2)

        if self.beam_shape == "Flat":
            mask = np.zeros(shape)
            mask[abs(self.x_offset_beam - x_positions) < self.l_beam / 2] = 1
            # Does this caculte the fractionn of the wire which is illumintated?
            # should  not be relevant for a power density f
            q = mask # / sum(mask) # i think not dividing is correct (Chr)
            # calculate fraction area which prijected wire makes up
            # Area wire/area beam
            q *= (self.d_wire * self.l_beam) / (np.pi * (self.l_beam/2)**2)

        elif self.beam_shape == "Gaussian":
            q = (
                1
                / (2 * np.pi * self.sigma_beam**2)
                * np.exp(
                    -((x_positions - self.x_offset_beam) ** 2)
                    / (2 * (self.sigma_beam) ** 2)
                )
                #* self.l_segment # multiplying by l_seg happens in delta_T
                * self.d_wire
            )
            # q *= (self.d_wire * self.sigma_beam) / (np.pi * self.sigma_beam**2)

        elif self.beam_shape == "Point":
            mask = np.zeros(shape)
            mask[np.abs((self.x_offset_beam - x_positions)) < (self.l_segment / 2)] = 1
            q = mask

        elif self.beam_shape == "Tschersich":
            q = (
                self.j_norm_linear(x = x_positions - self.x_offset_beam)
                # * self.l_segment # multiplying by l_seg happens in delta_T
                * self.d_wire)
        else:
            raise Exception("Unrecognized beam shape")

        # q is the fraction of particles which hit in this case
        # E_rec is for 2 atoms
        return self.phi_beam * self.E_recombination * (1/2) * q

    def f_laser(self) -> np.ndarray:
        """Deprecated."""
        print("Deprecated. Use f_beam instead.")
        shape = len(self.T_distribution)
        x_positions = (np.arange(len(self.T_distribution)) + 0.5) * self.l_segment - (
            self.l_wire / 2
        )

        if self.beam_shape == "Flat":
            # flat circular profile
            mask = np.zeros(shape)
            mask[
                (
                    ((-self.l_beam / 2) + self.x_offset_beam < x_positions)
                    & (x_positions < (self.l_beam / 2) + self.x_offset_beam)
                )
            ] = 1
            return mask * (
                (self.emissivity * self.p_laser * self.d_wire)
                / (np.pi * (self.l_beam / 2) ** 2)
            )

        elif self.beam_shape == "Gaussian":
            return (
                (self.emissivity * self.p_laser * self.d_wire)
                * (1 / (2 * np.pi * self.sigma_beam**2))
                * np.exp(
                    (-1 / 2)
                    * ((x_positions - self.x_offset_beam) / self.sigma_beam) ** 2
                )
            )

        elif self.beam_shape == "Point":
            mask = np.zeros(shape)
            mask[np.abs((self.x_offset_beam - x_positions)) < (self.l_segment / 2)] = 1
            return self.emissivity * self.p_laser / self.l_segment

        else:
            raise Exception("Unrecognized beam shape")

    def f_bb(self) -> np.ndarray:
        """Heat absorbed from the the black body radiation of the cracker."""
        x_positions = (np.arange(len(self.T_distribution)) + 0.5) * self.l_segment - (
            self.l_wire / 2
        )
        A_sphere = 4 * np.pi * self.dist_cracker_wire**2
        A_incident = self.l_beam * self.d_wire
        mask = np.zeros(len(x_positions))
        mask[
            (
                ((-self.l_beam / 2) + self.x_offset_beam < x_positions)
                & (x_positions < (self.l_beam / 2) + self.x_offset_beam)
            )
        ] = 1

        f = (
            mask
            * self.emissivity  # Note: Emissivity = absorbtivity of wire
            * self.sigma_stefan_boltzmann
            * (self.T_cracker**4 - self.T_background**4)
            * self.A_cracker
            * (A_incident / A_sphere)
            / self.l_beam # why is this here? CHr 2024-04-19
                            # Ah I see it canceles the l_beam from A_incident
        )  # normalization to length density

        return f

    def f_beam_gas(self) -> np.ndarray:
        """Kinetic energy transfer from the gas molecules and atoms of the beam."""
        n_atoms = self.f_beam() / (self.E_recombination / 2)
        n_molecules = (n_atoms / 2) * (1 / self.crack_eff - 1)
        f_atoms = (
            n_atoms * (3 / 2) * self.k_boltzmann * (self.T_atoms - self.T_distribution)
        )
        f_molecules = (
            n_molecules
            * (3 / 2)
            * self.k_boltzmann
            * (self.T_molecules - self.T_distribution)
        )
        f = f_atoms + f_molecules
        return f

    def f_background_gas(self) -> np.ndarray:
        """
        Kinetic energy transfer of residual background gas.
        Assumes background gas is primarily due to beam gas and all components
          have the same pumping speeds.
        """
        m = self.m_molecular_gas

        q = (
            (self.pressure / 4)
            * np.sqrt(3 * self.k_boltzmann / m)
            * (self.T_distribution - self.T_background)
            / np.sqrt(self.T_background)
        )

        # multiply with total outward surface area of wire segment
        f = q * np.pi * self.d_wire * self.l_segment / self.l_segment

        return f

    def temperature_change(self, time_step: float) -> np.ndarray:
        """
        Compute temperature change of single time step.
        Raises an error, if the temperature distribution does not converge. In this case a smaller time step in the simulation is necessary.
        """
        delta_T = (
            (
                self.f_el()
                - self.f_rad()
                - self.f_conduction()
                + self.f_beam()
                + self.f_beam_gas()
                + self.f_bb()
                - self.f_background_gas()
                # + self.f_laser()
            )
            * self.l_segment
            * time_step
        ) / (
            self.density * self.A_cross_section * self.l_segment * self.c_specific_heat
        )
        if not np.all(np.isfinite(delta_T)):
            raise ValueError(
                "The temerature distribution diverges. Use smaller time steps."
            )

        return delta_T

    def simulation_step(self, time_step: float) -> None:
        """Compute temperature after single simulation step."""
        self.T_distribution = self.T_distribution + self.temperature_change(time_step)
        self.simulation_time = self.simulation_time + time_step

    def simulate(self, n_steps: int = 15000, record_steps: int = 150, time_step: float = 0.001) -> None:
        """ 
        Simulates temperature of wire segments.
        n_steps : number of simulation steps
        record_steps : number of steps to store the intermediate temperature distribtutions
        time_step : time step in seconds
        """
        self.record_dict = {}
        self.record_dict["T_distribution"] = np.zeros(
            (record_steps + 1, self.n_wire_elements)
        )
        self.record_dict["time"] = np.zeros(record_steps + 1)

        # Intial State of the Wire
        self.simulation_time = 0
        if self.T_base is None:
            self.T_distribution = np.asarray(
                [self.T_background for n in range(self.n_wire_elements)]
            )
        else:
            self.T_distribution = self.T_base

        for i in range(n_steps + 1):
            if i in range(0, n_steps + 1, (n_steps // record_steps)):
                self.record_dict["T_distribution"][
                    i // ((n_steps // record_steps))
                ] = self.T_distribution
                self.record_dict["time"][
                    i // ((n_steps // record_steps))
                ] = self.simulation_time
            self.simulation_step(time_step=time_step)

    def save(self, filename: str) -> None:
        """ Save this object as pickled file. """
        with open(filename + ".pkl", "wb") as f:
            dill.dump(self, f)

    def load(self, filename):
        with open(filename + ".pkl", "rb") as f:
            wire = dill.load(f)
        return wire

    def U_wire(self, i: int) -> float:
        """ Calculate volatage drop over ith wire element. """
        U = (
            self.resistance_total(self.record_dict["T_distribution"][i])
            * self.i_current
        )
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
                 
        ax1.set_ylabel("Temperature [°C]")
        ax1.set_xlabel(r"Position Along Wire [mm]")
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

    def plot_T_final(self, filename="plots/T_final"):
        # Plot Temperature over Wire for start and end of simulation
        # Plot Temperature over Wire
        plt.figure(0, figsize=(8,6.5))
        ax1 = plt.gca()

        x_lst = [1000 * ((i + 0.5) * self.l_segment - (self.l_wire / 2))
                for i in range(self.n_wire_elements)]
        T_beam_on = self.record_dict["T_distribution"][-1]
        T_lst = [T_beam_on]

        R_arr = np.zeros(2)
        for i,T_dist in enumerate(T_lst):
            self.T_distribution = T_dist
            R_arr[i] = self.resistance_total()
        

        ax1.plot(x_lst, T_lst[0] - 273.15, "-", label=r"T(x) in equilibrium" 
                 #+ "R = {:.3f}".format(R_arr[0]) + r"$\Omega$"
                 ,
                 linewidth = 3
                 )

                 
        ax1.set_ylabel("Temperature [°C]")
        ax1.set_xlabel(r"Position Along Wire [mm]")
        # plt.title(r"$d_{wire}$ = " + "{}".format(self.d_wire * 10**6) 
        #           + r"$\mu m$" +", I = " + "{}".format(self.i_current * 10**3)
        #           + r"$mA$" + r", $\phi_{beam}$ = 10^" + "{:.2f}".format(
        #           np.log10(self.phi_beam)))
        plt.grid(True)
        # get existing handles and labels
        handles, labels = plt.gca().get_legend_handles_labels()
        plt.legend(handles, labels, shadow=True)
        
        plt.tight_layout()
        format_im = 'png' #'pdf' or png
        dpi = 300
        plt.savefig(filename + '.{}'.format(format_im),
                    format=format_im, dpi=dpi)
        ax1.cla()
        return

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
        
        f_el_arr = self.f_el()
        f_conduction_arr = self.f_conduction()
        f_rad_arr = self.f_rad()
        f_beam_arr = self.f_beam()
        f_beam_gas_arr = self.f_beam_gas()
        f_bb_arr = self.f_bb()
        f_background_gas_arr = self.f_background_gas()
        # f_laser_arr = self.f_laser()

        # Plot endstate of heat flow
        plt.figure(0, figsize=(8,6.5))
        ax1=plt.gca()

        # if self.bodge == True:
        #     for i in range(0,25):
        #         f_el_arr[i] = 0
        #     for i in range(self.n_wire_elements - 25,self.n_wire_elements):
        #         f_el_arr[i] = 0
        ax1.plot(x_lst, f_el_arr, "-", label=r"$f_{el}$")
        #bodge_start
        try:
            if self.bodge == True:
                ax1.plot(x_lst, f_conduction_bodge_arr, linestyle = (0, (1, 1))
                        , label=r"$f_{\mathrm{cond. piecewise}}$")
            else:
                ax1.plot(x_lst, f_conduction_arr, linestyle = (0, (1, 1)),
                         label=r"$f_{conduction}$")
        except:
            #bodge_end
            ax1.plot(x_lst, f_conduction_arr, linestyle = (0, (1, 1)), label=r"$f_{conduction}$")
        ax1.plot(x_lst, f_rad_arr, linestyle = (0, (1, 1)), label=r"$f_{rad}$")
        ax1.plot(x_lst, f_beam_arr, "-", label=r"$f_{rec}$")
        ax1.plot(x_lst, f_beam_gas_arr, "-", label=r"$f_{beam \,gas}$")
        ax1.plot(x_lst, f_bb_arr, "-", label=r"$f_{bb}$")
        ax1.plot(x_lst, f_background_gas_arr, linestyle = (0, (1, 1))
                 , label=r"$f_{bkgd \, gas}$")
        # ax1.plot(x_lst, f_laser_arr, "-"
        #          , label=r"$f_{laser}$")

        ax1.set_ylabel("Heat Flow [W/m]", fontsize = 16)
        ax1.set_xlabel(r"Position Along Wire [mm]", fontsize = 16)
        ax1.tick_params(axis='both', which='major', labelsize=12)

        #change line width after the fact
        for line in plt.gca().lines:
             line.set_linewidth(4.)

        plt.grid(True)
        plt.legend(shadow=True)



        #fancy legend:
        if True:
            h, l = ax1.get_legend_handles_labels()
            sources = [0,5,3,
                        4
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
               shadow = True,
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
               shadow = True,
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

    # def plot_signal(self, ax=None):
    #     """ Plot Temperature over Wire for start and end of simulation. """

    #     if ax is None:
    #         fig = plt.figure(0, figsize=(8, 6.5))
    #         ax = plt.gca()

    #     x_lst = [
    #         1000 * ((i + 0.5) * self.l_segment - (self.l_wire / 2))
    #         for i in range(self.n_wire_elements)
    #     ]
    #     T_beam_off = self.record_dict["T_distribution"][0]
    #     T_beam_on = self.record_dict["T_distribution"][-1]
    #     T_lst = [T_beam_off, T_beam_on]

    #     R_arr = np.zeros(2)
    #     for i, T_dist in enumerate(T_lst):
    #         R_arr[i] = self.resistance_total(T_dist)

    #     U_delta = (R_arr[1] - R_arr[0]) * self.i_current
    #     signal = (R_arr[1] - R_arr[0]) / R_arr[0]

    #     ax.plot(
    #         x_lst,
    #         T_lst[0] - 273.15,
    #         "-",
    #         label=r"Beam Off, " + "R = {:.3f}".format(R_arr[0]) + r"$\Omega$",
    #     )
    #     ax.plot(
    #         x_lst,
    #         T_lst[1] - 273.15,
    #         "-",
    #         label=r"Beam On, " + "R = {:.3f}".format(R_arr[1]) + r"$\Omega$",
    #     )

    #     ax.set_ylabel("Temperature [°C]")
    #     ax.set_xlabel(r"Position Along Wire [mm]")
    #     ax.set_title(
    #         r"$d_{wire}$ = "
    #         + "{}".format(self.d_wire * 10**6)
    #         + r"$\mu m$"
    #         + ", I = "
    #         + "{}".format(self.i_current * 10**3)
    #         + r"$mA$"
    #         + r", $\phi_{beam}$ = 10^"
    #         + "{:.2f}".format(np.log10(self.phi_beam))
    #     )
    #     ax.grid(True)
    #     # get existing handles and labels
    #     handles, labels = ax.get_legend_handles_labels()
    #     # create a patch with no color
    #     empty_patch = mpatches.Patch(color="none", label="Extra label")
    #     handles.append(empty_patch)
    #     labels.append(
    #         "Signal: {:.2%}, ".format(signal)
    #         + r"$\Delta U$"
    #         + " = {:.2f}".format(U_delta * 10**3)
    #         + " mV, "
    #     )
    #     ax.legend(handles, labels, shadow=True)
    #     return ax

    # def plot_R_over_t(self, ax=None):
    #     # Plot Resistance over time
    #     if ax is None:
    #         fig = plt.figure(0, figsize=(8, 6.5))
    #         ax = plt.gca()

    #     t_lst = self.record_dict["time"]
    #     steps = len(t_lst)
    #     R_lst = [
    #         self.resistance_total(self.record_dict["T_distribution"][i])
    #         for i in range(steps)
    #     ]

    #     R_tau = R_lst[0] + (R_lst[-1] - R_lst[0]) * (1 - 1 / np.exp(1))
    #     R_tau_lst = [R_tau for i in range(len(t_lst))]
    #     R_95 = R_lst[0] + (R_lst[-1] - R_lst[0]) * 0.95
    #     R_95_lst = [R_95 for i in range(len(t_lst))]
    #     # calculate time at which these are reached
    #     t_tau = t_lst[np.argmin(np.absolute(R_lst - R_tau))]
    #     t_95 = t_lst[np.argmin(np.absolute(R_lst - R_95))]

    #     ax.plot(t_lst, R_lst, "-", label="Resistance")
    #     ax.plot(
    #         t_lst,
    #         R_tau_lst,
    #         "-",
    #         label=r"$\Delta R \cdot$(1 - 1/e)" + ", t = {:.3f}".format(t_tau),
    #     )
    #     ax.plot(
    #         t_lst,
    #         R_95_lst,
    #         "-",
    #         label=r"$0.95 \cdot \Delta R$" + ", t = {:.3f}".format(t_95),
    #     )
    #     ax.set_ylabel(r"Resistance [$\Omega$]")
    #     ax.set_xlabel(r"time [s]")
    #     plt.grid(True)
    #     plt.legend(shadow=True)

    #     return ax

    # def plot_heat_flow(self, ax=None, log_y: bool = False):
    #     """ Calculate endstate of heat flow """
    #     x_lst = [
    #         1000 * ((i + 0.5) * self.l_segment - (self.l_wire / 2))
    #         for i in range(self.n_wire_elements)
    #     ]
    #     self.T_distribution = self.record_dict["T_distribution"][-1]
    #     f_el_arr = self.f_el()
    #     f_conduction_arr = self.f_conduction()
    #     f_rad_arr = self.f_rad()
    #     f_beam_arr = self.f_beam()
    #     f_beam_gas_arr = self.f_beam_gas()
    #     f_bb_arr = self.f_bb()
    #     f_background_gas_arr = self.f_background_gas()
    #     # f_laser_arr = self.f_laser()

    #     if ax is None:
    #         fig = plt.figure(0, figsize=(8, 6.5))
    #         ax = plt.gca()

    #     ax.plot(x_lst, f_el_arr, "-", label=r"$F_{el}$")

    #     ax.plot(x_lst, f_conduction_arr, linestyle = (0, (1, 1)), label=r"$-F_{conduction}$")
    #     ax.plot(x_lst, f_rad_arr, linestyle = (0, (1, 1)), label=r"$-F_{rad}$")
    #     ax.plot(x_lst, f_beam_arr, "-", label=r"$F_{rec}$")
    #     ax.plot(x_lst, f_beam_gas_arr, "-", label=r"$F_{beam \,gas}$")
    #     ax.plot(x_lst, f_bb_arr, "-", label=r"$F_{bb\, cracker}$")
    #     ax.plot(x_lst, f_background_gas_arr, linestyle = (0, (1, 1)), label=r"$-F_{bkgd \, gas}$")
    #     # ax.plot(x_lst, f_laser_arr, "-", label=r"$F_{laser}$")

    #     ax.set_ylabel("Heat Flow [W/m]", fontsize=16)
    #     ax.set_xlabel(r"Position Along Wire [mm]", fontsize=16)
    #     ax.tick_params(axis="both", which="major", labelsize=12)
    #     ax.grid(True)
    #     ax.legend(shadow=True)

    #     # fancy legend:
    #     if True:
    #         h, l = ax.get_legend_handles_labels()
    #         sources = [0, 3, 4, 5, 7]
    #         sinks = [1, 2, 6]

    #         l1 = ax.legend(
    #             [h[i] for i in sources],
    #             [l[i] for i in sources],
    #             # shadow = True,
    #             # framealpha = 0.5,
    #             loc="upper left",
    #             bbox_to_anchor=(1, 1),
    #             fontsize=14,
    #             title="Heat Sources:",
    #             title_fontsize=14,
    #             ncol=1,
    #         )
    #         ax.add_artist(l1)
    #         ax.legend(
    #             [h[i] for i in sinks],
    #             [l[i] for i in sinks],
    #             # shadow = True,
    #             # framealpha = 0.5,
    #             loc="upper left",
    #             bbox_to_anchor=(1, 0.60),
    #             fontsize=14,
    #             title="Heat Sinks:",
    #             title_fontsize=14,
    #             ncol=1,
    #         )

    #     if log_y == True:
    #         ax.set_yscale("log")

    #     return ax
