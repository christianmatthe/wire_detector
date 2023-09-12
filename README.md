# wire_detector

Simulation of a wire detector for atomic hydrogen

Note the `Wire_detector` package is only locally installable. 

Install Instructions: Type the following into a console

- `git clone https://github.com/christianmatthe/wire_detector` to copy this repository.

- Use `cd wire_detector` to navigate to the wire_detector folder containing `setup.py`

- Use `pip install .` to install wire detector as a local package.

You can now import `Wire_detector` or just import the Wire Class via  `from Wire_detector import Wire` from anywhere on your system.

## Simulation of Wire Temperature

The temperature distribution of the wire is simulated with `wire_detector.Wire`. The wire is divided into `Wire.n_wire_elements` segments of equal length and the heat flow between these elements is computed with the following equation (for each time step):

$$\Delta f_\mathrm{tot} = f_\mathrm{el} + f_\mathrm{beam} + f_\mathrm{beam\_gas} + f_\mathrm{bb}  + f_\mathrm{laser} - f_\mathrm{ rad} - f_\mathrm{conduction}-f_\mathrm{background\_gas}$$

This is the heat flow per unit length ($[f]=\rm \frac {Power}{Length}$).

- $f_\mathrm{el}$: Heat due to electrical current.
    
    $$f_\mathrm{el} = I^2 *R(T_i) / l$$
    
    where $I$ is the current, $R(T_i) = \rho\cdot l/A \cdot (1+ A)\cdot(T_i-T_0)$ is the resistance of the segment $i$ and $l$ is the length of the segment. $A$ denotes the cross-sectional area of the wire.
    
- $f_\mathrm{rad}$: Heat transfer through radiation (emission and absorption of surrounding).
    
    $$f_\mathrm{rad} = \pi\cdot  d_\mathrm{wire}\cdot  \sigma\cdot \epsilon \cdot (T_i^4 -T_\mathrm{background}^4)$$
    
    where $d$ is the diameter of the wire, $\sigma$ is the Stephan-Boltzmann-constant and $\epsilon$ is the emissivity. $\pi\cdot d\cdot l$ is the area, and divided by $l$ just results in $\pi \cdot d$.
    
- $f_\mathrm{conduction}$: Heat transfer through conduction
    
    $$f_\mathrm{conduction} =k(T_i)\cdot  \frac {( T_{i-1}-T_i)+(T_{i+1}-T_i)}{l}\cdot \frac Al$$
    
    where $k$ is the heat conductivity coefficient and $A$ is the cross sectional area of the wire. This is the second-order central difference approximation.
    
- $f_\mathrm{background\_gas}$: Heat transfer due to interactions with the surrounding gas.
    
    $$f_\mathrm{background\_gas} = \frac {\pi d\cdot l}{l}\frac p4 \sqrt{\frac {3\cdot k_B}m\frac {(T_i - T_\mathrm{background})^2}{T_\mathrm{background}}}$$ 
    
- $f_\mathrm{beam\_gas}$:  The heat transfer from the atoms and molecules of the gas.

    $$f_\mathrm{beam\_gas} = f_\mathrm{at.} + f_\mathrm{mol.}$$

    $$f_\mathrm{at./mol.} = \frac 32 n_\mathrm{at./mol.} k_B\cdot  (T_\mathrm{at./mol.} - T_i)$$


- $f_\mathrm{beam}$: Heat transfer due to the hydrogen beam source.
    
    $$f_\mathrm{beam} = b(i)\cdot\frac { 2\phi \cdot d_\mathrm{wire}\cdot E_\mathrm{recomb.} }{\pi l ^2_\mathrm {beam}}$$
    
    where $b$ is the shape function of the beam (flat or gaussian), $\phi$ is the beam intensity (atoms/second) and $l_\mathrm{beam}$ is the diameter or (standard deviation) of the beam.
    
- $f_\mathrm{bb}$: Blackbody radiation from cracker
    
    $$f_\mathrm{bb} = \varepsilon \cdot \sigma \cdot T_\mathrm{cracker}^4\cdot A_\mathrm{cracker}\frac {l\cdot d_\mathrm{wire}}{4\pi d_\mathrm{cracker}^2}\frac 1 l$$
    
    where $\sigma$ is the Stephan-Boltzmann-constant and $\epsilon$ is the emissivity and $A_\mathrm{cracker}$ is the area of the cracker visible to the wire and the fraction gives the solid angle.
    
- $f_\mathrm{laser}$: Heat transfer specifically due to a laser.
    
    see $f_\mathrm{beam}$

To obtain the temperature change from the heat flow:
$$\Delta T = \Delta f_\mathrm{tot} \cdot \frac {l \cdot \Delta T}{\rho \cdot A \cdot l\cdot c}$$
where $\rho$ is the density of the wire material, $A$ is the cross section, $l$ is the length of one segment and $c$ is the specific heat of the material. 