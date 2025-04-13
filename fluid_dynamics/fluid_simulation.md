## The Fluid Sloshing Simulation

This simulation models two-dimensional sloshing of a fluid in a rectangular tank.

The sloshing behavior is modeled using a superposition of natural oscillation modes, each with its own natural frequency.

- The blue line shows the free surface elevation
- Light blue region represents the fluid body
- Velocity vectors (arrows) show the internal fluid motion
- Black rectangle represents the moving tank

### Physics Equations:

The core of this simulation is based on:

1. Natural Frequencies:

```
ω_n = √(g * k_n * tanh(k_n * h))
```

Where:

- g is gravity
- k\*n = nπ/L is the wave number
- h is the fluid depth.

2. Modal Amplitudes: The amplitude of each mode is calculated using the equation of a damped harmonic oscillator under forced excitation.

3. Surface Elevation: The free surface is modeled as the superposition of modal contributions:

```
η(x,t) = h + Σ A_n(t) * cos(k*n * x)
```

4. Velocity Field: Derived from a velocity potential that satisfies Laplace's equation with appropriate boundary conditions.

## Customize

You can modify these parameters to explore different sloshing conditions:

```python
sim = SloshingSimulation(
length=1.0, # Tank length (m)
height=0.6, # Tank height (m)
fill_level=0.3, # Fluid fill level (m)
excitation_amplitude=0.03, # Excitation amplitude (m)
excitation_frequency=2.0, # Excitation frequency (Hz)
damping=0.05, # Damping coefficient
modes=5, # Number of modes to include
)
```
