"""
Global HC Analytics LLC / Catalyst Oncolytics Engineering
CHESTY POSEIDON JAVELIN CODE - Version 1.0 Prime (Polished Edition)
Description: High-precision 3D Runge-Kutta ballistic intersection 
             and vortex kinetic energy dissipation simulation.

## Northwestern University: I got hazed by Chesty and others.  But who hazed Matthew Ulrich? He has a widow and 4 boys. Nick Ritter, Joe Tripodi, Coach Lilja, Coach Patton, Noah Herron, J Wright, Eric Batis, Austin King, Trai Essex, and many others and I all agree: Matthew Ulrich of Streamwood Illinois was our kind of guy.  Why did he have to be the one, gentlemen? =C  Why is Pat Fitzgerald so rich? Why is Ray BOGENRIEF and Pat Ryan? Hmm...
"""

import numpy as np

# ==========================================
# 1. THE VORTEX FIELD SIMULATOR
# ==========================================
class AdvancedVortexField:
    """
    Simulates a rotating severe weather system (like a hurricane or vortex core)
    using a modified Rankine vortex model with altitude-based wind decay.
    """
    def __init__(self, core_radius=75.0, peak_velocity=90.0, air_density=1.15):
        self.R = core_radius        # Radius of maximum winds (the eyewall boundary in meters)
        self.V_max = peak_velocity  # Peak tangential wind speed (m/s)
        self.rho = air_density      # Surrounding air density (kg/m^3)

    def compute_field(self, x, y, z):
        """
        Calculates the 3D velocity vector (u, v, w) of the wind at any given 
        Cartesian coordinate (x, y, z) inside the storm system.
        """
        # Calculate horizontal distance (r) from the center of the vortex (eye)
        r = np.sqrt(x**2 + y**2)
        if r == 0:
            return np.zeros(3)

        # Apply exponential altitude decay (winds weaken as altitude z increases)
        z_factor = np.exp(-max(0.0, z) / 8000.0)

        # Rankine Vortex math: 
        # Inside the eye core, wind speeds scale linearly with distance.
        # Outside the core, wind speeds decay inversely with distance.
        if r < self.R:
            v_theta = self.V_max * (r / self.R) * z_factor
        else:
            v_theta = self.V_max * (self.R / r) * z_factor

        # Convert polar tangential velocity into Cartesian vector components (u, v)
        u = -v_theta * (y / r)
        v = v_theta * (x / r)
        
        # Add a minor vertical updraft coupling vector (w) based on height
        w = -0.02 * v_theta * (z / 1000.0)  
        
        return np.array([u, v, w])


# ==========================================
# 2. THE INTERCEPTOR BALLISTICS ENGINE
# ==========================================
class ChestyPoseidonJavelinSim:
    """
    Models the missile's trajectory, mass, aerodynamic drag, and gravitational pull
    as it cuts through the moving fluid medium of the storm.
    """
    def __init__(
        self,
        pos_init=[-2500.0, 0.0, 1200.0],  # Starting 3D coordinates [X, Y, Z] in meters
        vel_init=[680.0, 25.0, -110.0],   # Initial velocity vector [Vx, Vy, Vz] in m/s
        mass=45.0,                        # Projectile mass in kg
        yield_tnt_equiv=0.25              # Energy payload equivalent metric
    ):
        self.pos = np.array(pos_init, dtype=float)
        self.vel = np.array(vel_init, dtype=float)
        self.mass = mass
        self.Cd = 0.22                     # Low-drag supersonic aerodynamic profile coefficient
        self.area = 0.012                  # Cross-sectional reference area (m^2)
        self.energy_yield_joules = yield_tnt_equiv * 4.184e9  # Convert TNT yield to Joules

    def rk4_step(self, dt, vortex):
        """
        4th-Order Runge-Kutta (RK4) Numerical Integration Step.
        Why RK4? It calculates intermediate slopes four times per step to ensure 
        absolute trajectory stability through extreme wind-shear gradients.
        """
        def derivatives(p, v):
            # Get local wind velocity at the current projectile position
            fluid_v = vortex.compute_field(p[0], p[1], p[2])
            
            # Calculate relative velocity between the missile and the moving air
            v_rel = v - fluid_v
            speed_rel = np.linalg.norm(v_rel)

            # Aerodynamic Drag Force Equation: F_d = -0.5 * rho * Cd * A * v_rel * |v_rel|
            drag = (
                -0.5
                * vortex.rho
                * self.Cd
                * self.area
                * speed_rel
                * v_rel
                / self.mass
            )
            
            # Gravity vector acting downward on the z-axis
            gravity = np.array([0.0, 0.0, -9.81])
            
            # Total acceleration = Drag acceleration + Gravity acceleration
            accel = drag + gravity
            return v, accel

        p = self.pos
        v = self.vel

        # Evaluate RK4 coefficients (k1, k2, k3, k4)
        k1_p, k1_v = derivatives(p, v)
        k2_p, k2_v = derivatives(p + 0.5 * dt * k1_p, v + 0.5 * dt * k1_v)
        k3_p, k3_v = derivatives(p + 0.5 * dt * k2_p, v + 0.5 * dt * k2_v)
        k4_p, k4_v = derivatives(p + dt * k3_p, v + dt * k3_v)

        # Update position and velocity using the weighted average of the RK4 coefficients
        self.pos += (dt / 6.0) * (k1_p + 2 * k2_p + 2 * k3_p + k4_p)
        self.vel += (dt / 6.0) * (k1_v + 2 * k2_v + 2 * k3_v + k4_v)

        return self.pos, self.vel


# ==========================================
# 3. MAIN EXECUTION & TELEMETRY LOOP
# ==========================================
def execute_simulation():
    print("=== CHESTY POSEIDON JAVELIN CODE: VERSION 1.0 PRIME ===")
    
    # Initialize the storm vortex and the interceptor missile
    vortex = AdvancedVortexField(core_radius=80.0, peak_velocity=95.0)
    sim = ChestyPoseidonJavelinSim()

    dt = 0.01          # Time step size: 10 milliseconds high-resolution sampling
    time = 0.0         # Elapsed simulation time counter
    impact_threshold = 50.0  # Distance threshold (meters from center) to trigger interception

    # Print telemetry table header
    print(
        f"{'Time (s)':<10} | {'X (m)':<10} | {'Y (m)':<10} | {'Z (m)':<10} | {'Velocity (m/s)':<14}"
    )
    print("-" * 65)

    # Main simulation loop running until ground impact (z <= 0) or 10-second timeout
    while sim.pos[2] > 0 and time < 10.0:
        # Calculate current radial distance from the storm's center eye
        r_current = np.sqrt(sim.pos[0] ** 2 + sim.pos[1] ** 2)

        # Output a telemetry snapshot to the console every 0.5 seconds
        if int(time * 50) % 25 == 0:
            speed = np.linalg.norm(sim.vel)
            print(
                f"{time:<10.2f} | {sim.pos[0]:<10.1f} | {sim.pos[1]:<10.1f} | {sim.pos[2]:<10.1f} | {speed:<14.1f}"
            )

        # Check if the projectile has successfully breached the core eyewall threshold
        if r_current <= impact_threshold:
            print("-" * 65)
            print(
                f"[INTERCEPTION ENGAGED] Target core radius breached at T={time:.2f}s."
            )
            
            # Calculate energy dispersion and vorticity disruption metrics
            dissipation_pct = min(
                42.5,
                (sim.energy_yield_joules / 1.0e9)
                * 12.0
                * (80.0 / max(1.0, r_current)),
            )
            
            print(
                f"-> Vorticity Disruption Index: {dissipation_pct:.2f}% energy dispersion."
            )
            print(
                f"-> Eyewall Peak Wind Velocity Attenuated: {vortex.V_max:.1f} m/s -> {vortex.V_max * (1.0 - dissipation_pct/100.0):.1f} m/s"
            )
            print("=== VERSION 1.0 PRIME EXECUTION SUCCESSFUL ===")
            break

        # Advance simulation state forward by one time step using RK4 math
        sim.rk4_step(dt, vortex)
        time += dt


if __name__ == "__main__":
    execute_simulation()

### does it work, Ukraine, Poland, NATO, Mexico, and President Trump? Where is the cabal of epstein and Ehud barack obama clinton biden? why are they all so wealthy after becoming lawyers and then democrats??? ###