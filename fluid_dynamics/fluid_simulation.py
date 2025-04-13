import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches


class SloshingSimulation:
    def __init__(
        self,
        length=1.0,
        height=0.5,
        fill_level=0.3,
        excitation_amplitude=0.05,
        excitation_frequency=3.0,
        damping=0.05,
        modes=5,
        g=9.81,
    ):
        """
        Initialize a 2D sloshing simulation in a rectangular tank

        Parameters:
        -----------
        length : float
            Length of the tank (m)
        height : float
            Height of the tank (m)
        fill_level : float
            Initial fill level of the fluid (m)
        excitation_amplitude : float
            Amplitude of the external excitation (m)
        excitation_frequency : float
            Frequency of the external excitation (Hz)
        damping : float
            Damping coefficient
        modes : int
            Number of modes to include in the simulation
        g : float
            Gravitational acceleration (m/s²)
        """
        self.L = length
        self.H = height
        self.h = fill_level
        self.A = excitation_amplitude
        self.omega = 2 * np.pi * excitation_frequency
        self.damping = damping
        self.modes = modes
        self.g = g

        # Calculate natural frequencies for each mode
        self.natural_frequencies = self._calculate_natural_frequencies()

    def _calculate_natural_frequencies(self):
        """Calculate natural frequencies for each sloshing mode"""
        n = np.arange(1, self.modes + 1)
        k_n = n * np.pi / self.L
        omega_n = np.sqrt(self.g * k_n * np.tanh(k_n * self.h))
        return omega_n

    def modal_amplitude(self, mode, t):
        """Calculate the modal amplitude for a given mode and time"""
        n = mode + 1  # Convert 0-indexed to 1-indexed
        k_n = n * np.pi / self.L
        omega_n = self.natural_frequencies[mode]

        # Modal response using a second-order ODE with damping
        # Simplification: use analytical solution for forced oscillation
        numerator = self.A * k_n
        denominator = (omega_n**2 - self.omega**2) ** 2 + (
            2 * self.damping * omega_n * self.omega
        ) ** 2
        amplitude = numerator / np.sqrt(denominator)

        # Phase angle
        phi = np.arctan2(
            2 * self.damping * omega_n * self.omega, omega_n**2 - self.omega**2
        )

        return amplitude * np.sin(self.omega * t - phi)

    def surface_elevation(self, x, t):
        """Calculate the surface elevation at position x and time t"""
        elevation = 0
        for mode in range(self.modes):
            n = mode + 1
            k_n = n * np.pi / self.L
            elevation += self.modal_amplitude(mode, t) * np.cos(k_n * x)

        # Add tank motion to surface elevation
        tank_motion = self.A * np.sin(self.omega * t)

        return self.h + elevation, tank_motion

    def velocity_field(self, x, y, t):
        """Calculate the velocity field at position (x, y) and time t"""
        u = 0  # x-component
        v = 0  # y-component

        for mode in range(self.modes):
            n = mode + 1
            k_n = n * np.pi / self.L
            amp = self.modal_amplitude(mode, t)

            # Velocity components derived from potential flow
            u += (
                amp
                * self.g
                * k_n
                * np.sinh(k_n * y)
                / np.cosh(k_n * self.h)
                * np.sin(k_n * x)
                / self.omega
            )
            v += (
                amp
                * self.g
                * k_n
                * np.cosh(k_n * y)
                / np.cosh(k_n * self.h)
                * np.cos(k_n * x)
                / self.omega
            )

        # Add tank motion to x-velocity
        u += self.A * self.omega * np.cos(self.omega * t)

        return u, v

    def run_simulation(self, duration=10, fps=30, show_velocity=True):
        """
        Create an animation of the sloshing fluid

        Parameters:
        -----------
        duration : float
            Duration of the animation in seconds
        fps : int
            Frames per second
        show_velocity : bool
            Whether to show velocity field arrows
        """
        # Create the figure and axes
        fig, ax = plt.subplots(figsize=(10, 6))

        # Number of frames
        frames = int(duration * fps)

        # Time array
        times = np.linspace(0, duration, frames)

        # Space discretization
        nx = 100
        x = np.linspace(0, self.L, nx)

        # Calculate initial surface elevation
        y_surface = np.zeros_like(x)
        for i, x_val in enumerate(x):
            y_surface[i], _ = self.surface_elevation(x_val, 0)

        # Tank rectangle
        tank = patches.Rectangle(
            (0, 0), self.L, self.H, fill=False, edgecolor="black", linewidth=2
        )
        ax.add_patch(tank)

        # Surface line
        (surface_line,) = ax.plot(x, y_surface, "b-", linewidth=2)

        # Initial polygon coordinates
        polygon_x = np.append(x, [self.L, 0])
        polygon_y = np.append(y_surface, [0, 0])

        # Fluid polygon
        fluid = patches.Polygon(
            np.column_stack((polygon_x, polygon_y)),
            closed=True,
            color="lightblue",
            alpha=0.7,
        )
        ax.add_patch(fluid)

        # Velocity field
        if show_velocity:
            # Create grid for velocity vectors
            nx_vel, ny_vel = 10, 6
            x_mesh, y_mesh = np.meshgrid(
                np.linspace(0, self.L, nx_vel), np.linspace(0, self.h, ny_vel)
            )

            # Initialize velocity vectors
            u_init = np.zeros_like(x_mesh)
            v_init = np.zeros_like(y_mesh)

            # Create quiver plot with initial zeros
            quiver = ax.quiver(
                x_mesh,
                y_mesh,
                u_init,
                v_init,
                scale=20,
                width=0.003,
                color="blue",
                alpha=0.7,
            )

        # Set axis limits with padding for tank motion
        ax.set_xlim(-self.A - 0.1, self.L + self.A + 0.1)
        ax.set_ylim(-0.05, self.H + 0.05)

        # Labels and title
        ax.set_xlabel("Length (m)")
        ax.set_ylabel("Height (m)")
        title = ax.set_title("Fluid Sloshing in Rectangular Tank (t = 0.0 s)")

        def animate(frame):
            t = times[frame]

            # Update title
            title.set_text(f"Fluid Sloshing in Rectangular Tank (t = {t:.2f} s)")

            # Calculate tank motion
            _, tank_motion = self.surface_elevation(0, t)

            # Update tank position
            tank.set_xy((tank_motion, 0))

            # Calculate surface points
            x_pos = np.linspace(0, self.L, nx)
            y_surface = np.zeros_like(x_pos)

            for i, x_val in enumerate(x_pos):
                y_surface[i], _ = self.surface_elevation(x_val, t)

            # Update surface line
            x_plot = x_pos + tank_motion
            surface_line.set_data(x_plot, y_surface)

            # Update fluid polygon
            polygon_x = np.append(x_plot, [tank_motion + self.L, tank_motion])
            polygon_y = np.append(y_surface, [0, 0])
            fluid.set_xy(np.column_stack((polygon_x, polygon_y)))

            # Update velocity field if enabled
            if show_velocity:
                # Shift grid points by tank motion
                shifted_x = x_mesh.flatten() + tank_motion
                quiver.set_offsets(np.column_stack((shifted_x, y_mesh.flatten())))

                # Calculate velocity field
                u = np.zeros_like(x_mesh)
                v = np.zeros_like(y_mesh)

                for i in range(x_mesh.shape[0]):
                    for j in range(x_mesh.shape[1]):
                        # Get the x-position in tank coordinates
                        tank_x = x_mesh[i, j]

                        # Find the nearest surface height
                        idx = int(tank_x / self.L * (nx - 1))
                        idx = max(0, min(idx, nx - 1))

                        # Only calculate for points below the surface
                        if y_mesh[i, j] < y_surface[idx]:
                            u[i, j], v[i, j] = self.velocity_field(
                                tank_x, y_mesh[i, j], t
                            )

                # Update velocity vectors
                quiver.set_UVC(u.flatten(), v.flatten())

            return [surface_line, tank, fluid] + ([quiver] if show_velocity else [])

        # Create the animation
        anim = FuncAnimation(
            fig, animate, frames=frames, interval=1000 / fps, blit=False
        )

        plt.tight_layout()
        return fig, anim


# Example usage
if __name__ == "__main__":
    # Create simulation with custom parameters
    sim = SloshingSimulation(
        length=1.0,  # Tank length (m)
        height=0.6,  # Tank height (m)
        fill_level=0.3,  # Fluid fill level (m)
        excitation_amplitude=0.03,  # Excitation amplitude (m)
        excitation_frequency=2.0,  # Excitation frequency (Hz)
        damping=0.05,  # Damping coefficient
        modes=5,  # Number of modes to include
    )

    # Run animation
    fig, anim = sim.run_simulation(duration=10, fps=30, show_velocity=True)

    # Display the animation
    plt.show()

    # Uncomment to save animation as a video file (requires ffmpeg)
    # anim.save('sloshing_animation.mp4', writer='ffmpeg', fps=30, dpi=100)
