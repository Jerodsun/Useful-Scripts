## The Laminar Flow Simulation

This simulation models Poiseuille flow - a specific type of laminar flow that occurs between two parallel plates or in a pipe due to a pressure gradient.

1. **Parabolic Profile**: Velocity (indicated by color) is highest in the center of the channel and zero at the walls.

2. **Tracer Particle Movement**: Particles move faster in the center and slower near the walls, creating the characteristic curved paths in laminar flow.

3. **Flow Around Obstacles**: Streamlines smoothly curve around obstacles and the flow recovers downstream.

4. **Reynolds Number Effect**: Change the viscosity and velocity to alter the Reynolds number. At low Reynolds numbers, laminar flow dominates, while higher values might show departures from pure laminar behavior.

### Physics:

1. **Parabolic Velocity Profile**: In laminar channel flow, the velocity varies parabolically with height - zero at the walls and maximum at the center.

2. **Flow Visualization**:

   - Color mapping shows velocity magnitude (blue = slow, red = fast)
   - White streamlines indicate flow direction
   - Black dots act as tracer particles to visualize the flow patterns

3. **Optional Obstacles**: The simulation can include circular obstacles that disturb the flow field, allowing you to see how the laminar flow adapts around obstructions.

### Physics Equations:

The core is based on the analytical solution for Poiseuille flow:

```
u(y) = (1/2μ) * (dp/dx) * y * (H - y)
```

Where:

- u(y) is the velocity at height y
- μ is the fluid viscosity
- dp/dx is the pressure gradient
- H is the channel height

## Customize

You can modify these parameters to explore different laminar flow conditions:

```python
sim = LaminarFlowSimulation(
    channel_length=2.0,     # Length of the channel
    channel_height=0.5,     # Height of the channel
    max_velocity=1.0,       # Maximum velocity at the center
    viscosity=0.01,         # Fluid viscosity (affects Reynolds number)
    num_tracers=100,        # Number of tracer particles
    num_obstacles=3,        # Set to 0 for pure laminar flow, or add obstacles
    obstacle_radius=0.04    # Size of obstacles
)
```
