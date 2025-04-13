import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.colors as colors
from matplotlib.cm import ScalarMappable
from matplotlib import patches

class LaminarFlowSimulation:
    def __init__(self, 
                 channel_length=2.0, 
                 channel_height=0.5, 
                 max_velocity=1.0,
                 viscosity=0.01,
                 pressure_gradient=1.0,
                 num_tracers=50,
                 num_obstacles=0,
                 obstacle_radius=0.05):
        """
        Initialize a laminar flow simulation in a channel
        
        Parameters:
        -----------
        channel_length : float
            Length of the channel (m)
        channel_height : float
            Height of the channel (m)
        max_velocity : float
            Maximum velocity at the center of the channel (m/s)
        viscosity : float
            Kinematic viscosity of the fluid (m²/s)
        pressure_gradient : float
            Pressure gradient along the channel (Pa/m)
        num_tracers : int
            Number of tracer particles to simulate
        num_obstacles : int
            Number of circular obstacles to place in the flow
        obstacle_radius : float
            Radius of the obstacles (m)
        """
        self.L = channel_length
        self.H = channel_height
        self.U_max = max_velocity
        self.nu = viscosity
        self.dp_dx = pressure_gradient
        self.num_tracers = num_tracers
        self.num_obstacles = num_obstacles
        self.obstacle_radius = obstacle_radius
        
        # Generate initial tracer positions
        self.tracers = self._initialize_tracers()
        
        # Generate obstacles
        self.obstacles = self._initialize_obstacles()
    
    def _initialize_tracers(self):
        """Initialize tracer particles distributed throughout the channel"""
        tracers = []
        for _ in range(self.num_tracers):
            x = np.random.uniform(0, self.L)
            y = np.random.uniform(0.05, self.H - 0.05)
            tracers.append([x, y])
        return np.array(tracers)
    
    def _initialize_obstacles(self):
        """Initialize circular obstacles in the flow field"""
        obstacles = []
        if self.num_obstacles > 0:
            # Ensure obstacles don't overlap and stay within channel
            min_distance = 4 * self.obstacle_radius
            
            for _ in range(self.num_obstacles):
                while True:
                    # Propose a new obstacle position
                    x = np.random.uniform(0.2, self.L - 0.2)
                    y = np.random.uniform(2*self.obstacle_radius, self.H - 2*self.obstacle_radius)
                    new_obstacle = [x, y]
                    
                    # Check for overlap with existing obstacles
                    overlap = False
                    for existing in obstacles:
                        dist = np.sqrt((existing[0] - x)**2 + (existing[1] - y)**2)
                        if dist < min_distance:
                            overlap = True
                            break
                    
                    if not overlap:
                        obstacles.append(new_obstacle)
                        break
        
        return np.array(obstacles) if obstacles else np.empty((0, 2))
    
    def poiseuille_velocity(self, y):
        """
        Calculate the Poiseuille flow velocity at height y
        
        For a channel flow driven by pressure gradient, the velocity profile is:
        u(y) = (1/2μ) * (dp/dx) * y * (H - y)
        
        which is a parabolic profile with zero velocity at walls (y=0, y=H)
        """
        # Scale to match the specified max velocity
        scale_factor = self.U_max / (self.H**2 / 4)
        
        # Parabolic profile
        return scale_factor * y * (self.H - y)
    
    def get_velocity_at_point(self, x, y):
        """
        Get velocity vector at a specific point, accounting for obstacles
        """
        # Base Poiseuille flow velocity (x-component only in base flow)
        u = self.poiseuille_velocity(y)
        v = 0
        
        # Check if point is inside or very close to any obstacle
        for obs_x, obs_y in self.obstacles:
            dist = np.sqrt((x - obs_x)**2 + (y - obs_y)**2)
            
            # If inside or close to obstacle boundary, velocity is zero
            if dist <= self.obstacle_radius:
                return 0, 0
            
            # Simple approximation of flow around obstacle
            # Decrease velocity near obstacles and add a small deflection
            elif dist < 3 * self.obstacle_radius:
                # Reduce velocity near obstacle
                factor = (dist - self.obstacle_radius) / (2 * self.obstacle_radius)
                u *= factor
                
                # Add slight deflection around obstacle
                dx, dy = x - obs_x, y - obs_y
                angle = np.arctan2(dy, dx)
                v += 0.1 * u * np.sin(angle) / dist
        
        return u, v
    
    def update_tracers(self, dt):
        """Update the position of all tracer particles"""
        new_positions = []
        
        for x, y in self.tracers:
            # Get velocity at current position
            u, v = self.get_velocity_at_point(x, y)
            
            # Update position using simple Euler integration
            x_new = x + u * dt
            y_new = y + v * dt
            
            # Handle tracer particles that exit the domain
            if x_new > self.L:
                x_new = 0
                # Randomize y position for new tracers
                y_new = np.random.uniform(0.05, self.H - 0.05)
            
            # Ensure tracers stay within the channel height
            y_new = max(0.01, min(y_new, self.H - 0.01))
            
            new_positions.append([x_new, y_new])
        
        self.tracers = np.array(new_positions)
        return self.tracers
    
    def generate_velocity_field(self, nx=50, ny=20):
        """Generate velocity field data for visualization"""
        # Create grid points
        x = np.linspace(0, self.L, nx)
        y = np.linspace(0, self.H, ny)
        X, Y = np.meshgrid(x, y)
        
        # Calculate velocity at each grid point
        U = np.zeros_like(X)
        V = np.zeros_like(Y)
        speed = np.zeros_like(X)
        
        for i in range(ny):
            for j in range(nx):
                U[i, j], V[i, j] = self.get_velocity_at_point(X[i, j], Y[i, j])
                speed[i, j] = np.sqrt(U[i, j]**2 + V[i, j]**2)
        
        return X, Y, U, V, speed
    
    def run_simulation(self, duration=10, fps=30):
        """
        Run a visualization of the laminar flow simulation
        
        Parameters:
        -----------
        duration : float
            Duration of the simulation in seconds
        fps : int
            Frames per second for the animation
        """
        # Set up the figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Calculate time step and number of frames
        dt = 0.05  # Time step for simulation (independent of animation)
        num_frames = int(duration * fps)
        
        # Generate velocity field data for background coloring
        X, Y, U, V, speed = self.generate_velocity_field(nx=100, ny=50)
        
        # Create color mesh for velocity magnitude
        vmin, vmax = 0, self.U_max
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        
        # Plot the velocity field as a colormesh
        mesh = ax.pcolormesh(X, Y, speed, cmap='coolwarm', norm=norm, shading='gouraud')
        
        # Add a colorbar
        cbar = plt.colorbar(mesh, ax=ax)
        cbar.set_label('Velocity Magnitude (m/s)')
        
        # Add streamlines for flow visualization
        streamlines = ax.streamplot(X, Y, U, V, density=1, color='white', linewidth=0.7, arrowsize=0.5)
        
        # Plot channel boundaries
        ax.axhline(y=0, color='k', linestyle='-', linewidth=2)
        ax.axhline(y=self.H, color='k', linestyle='-', linewidth=2)
        
        # Initialize tracer particles plot
        tracers_plot, = ax.plot([], [], 'ko', markersize=3, alpha=0.7)
        
        # Add obstacle patches if any
        obstacle_patches = []
        for obs_x, obs_y in self.obstacles:
            circle = patches.Circle((obs_x, obs_y), self.obstacle_radius, 
                                    facecolor='gray', edgecolor='black', zorder=5)
            ax.add_patch(circle)
            obstacle_patches.append(circle)
        
        # Set axis properties
        ax.set_xlim(0, self.L)
        ax.set_ylim(0, self.H)
        ax.set_xlabel('Channel Length (m)')
        ax.set_ylabel('Channel Height (m)')
        ax.set_title('Laminar Flow in a Channel')
        ax.set_aspect('equal', adjustable='box')
        
        def animate(frame):
            # Update simulation time
            t = frame * dt
            
            # Update tracers
            self.update_tracers(dt)
            
            # Update tracer particle positions
            tracers_plot.set_data(self.tracers[:, 0], self.tracers[:, 1])
            
            return [tracers_plot] + obstacle_patches
        
        # Create animation
        anim = FuncAnimation(fig, animate, frames=num_frames, interval=1000/fps, blit=False)
        
        plt.tight_layout()
        return fig, anim

# Example usage
if __name__ == "__main__":
    # Create a basic laminar flow simulation
    sim = LaminarFlowSimulation(
        channel_length=2.0,  # Length of channel
        channel_height=0.5,  # Height of channel
        max_velocity=1.0,    # Maximum flow velocity
        viscosity=0.01,      # Fluid viscosity
        num_tracers=100,     # Number of tracer particles
        num_obstacles=3,     # Number of obstacles
        obstacle_radius=0.04  # Size of obstacles
    )
    
    # Run the simulation
    fig, anim = sim.run_simulation(duration=15, fps=30)
    
    # Display the animation
    plt.show()
    
    # Uncomment to save animation (requires ffmpeg)
    # anim.save('laminar_flow.mp4', writer='ffmpeg', fps=30, dpi=100)