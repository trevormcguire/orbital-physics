"""Physics Engine."""
import numpy as np

from core.physics import Coordinates, ObjectCollection, pairwise_accelerations

class SimulationEngine:
    """
    Engine to advance the orbital simulation forward in time.

    Attributes:
        objects (ObjectCollection): The collection of objects in the simulation.
        dt (float): Time step (in seconds).
        restitution (float): Coefficient of restitution for collisions (0 to 1).
        softening (float): Softening length to avoid singularities (in meters).
        history (dict): Stores positions of all objects at each step for plotting.
    """
    def __init__(self, objects: ObjectCollection, dt: float = 1.0, softening: float = 0.0, restitution: float = 1.0):
        self.objects = objects
        self.dt = float(dt)
        self.softening = float(softening)
        self.restitution = float(restitution)
        self.history = {obj.uuid: [obj.position().copy()] for obj in self.objects}
        # initial accelerations
        self.acc, self.last_potential = pairwise_accelerations(self.objects.objects, eps=self.softening)

    def step(self):
        dt = self.dt

        # 1) half-kick: v(t+dt/2) = v(t) + a(t)*dt/2
        for obj in self.objects:
            obj.velocity += 0.5 * dt * self.acc[obj.uuid]

        # 2) drift: r(t+dt) = r(t) + v(t+dt/2)*dt
        for obj in self.objects:
            new_pos = obj.position() + obj.velocity * dt
            obj.coordinates = Coordinates.from_iterable(new_pos)

        # 3) recompute accelerations at new positions
        self.acc, self.last_potential = pairwise_accelerations(self.objects.objects, eps=self.softening)

        # 4) half-kick: v(t+dt) = v(t+dt/2) + a(t+dt)*dt/2
        for obj in self.objects:
            obj.velocity += 0.5 * dt * self.acc[obj.uuid]

        # Collisions (after completing the symplectic step)
        self.objects.handle_collisions(restitution=self.restitution)

        # Record
        for obj in self.objects:
            self.history[obj.uuid].append(obj.position().copy())

    def run(self, steps: int):
        for _ in range(int(steps)):
            self.step()

    # Diagnostics
    def total_energy(self):
        K = 0.0
        for obj in self.objects:
            v2 = float(obj.velocity @ obj.velocity)
            K += 0.5 * obj.mass * v2
            # add spin KE if you want:
            # K += 0.5 * obj.moi * float(obj.angular_velocity @ obj.angular_velocity)
        U = self.last_potential  # set in last acceleration build
        return K + U

    def angular_momentum(self):
        L = np.zeros(3)
        for obj in self.objects:
            r = obj.position()
            p = obj.mass * obj.velocity
            L += np.cross(r, p)
            # add spin angular momentum if modeling rigid bodies: L += I·ω in body frame
        return L


def run_simulation(engine: SimulationEngine, steps: int, print_every: int = 100):
    E0 = engine.total_energy()
    L0 = engine.angular_momentum()
    for s in range(steps):
        engine.step()
        if s % print_every == 0:
            E = engine.total_energy()
            L = engine.angular_momentum()
            dE = (E - E0) / abs(E0)
            dL = np.linalg.norm(L - L0) / (np.linalg.norm(L0) + 1e-30)
            print(f"step {s}: ΔE={dE:.3e}, ΔL={dL:.3e}")
