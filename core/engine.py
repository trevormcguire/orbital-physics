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
    def __init__(self, objects: ObjectCollection, dt: float = 1.0, softening: float = 0.0, restitution: float = 1.0, max_hist: int = -1):
        self.objects = objects
        self.dt = float(dt)
        self.softening = float(softening)
        self.restitution = float(restitution)
        self.history = {obj.uuid: [obj.position().copy().tolist()] for obj in self.objects}
        self.max_hist = max_hist
        # initial accelerations
        self.acc, self.last_potential = pairwise_accelerations(self.objects.objects, eps=self.softening)
        self.time_elapsed = float(dt)

    def named_history(self, limit: int = 0):
        """Return history with object names as keys instead of UUIDs."""
        if limit > 0:
            return {obj.name: self.history[obj.uuid][-limit:] for obj in self.objects}
        return {obj.name: self.history[obj.uuid] for obj in self.objects}

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
        use_queue = self.max_hist is not None
        for obj in self.objects:
            self.history[obj.uuid].append(obj.position().copy().tolist())
            if use_queue and (len(self.history[obj.uuid]) > self.max_hist):
                self.history[obj.uuid].pop(0)

        self.time_elapsed += dt

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
    
    # def save_state(self) -> dict:
    #     """Return a JSON-serializable snapshot of the current state."""
    #     return {
    #         "time_elapsed": self.time_elapsed,
    #         "objects": [obj.to_dict() for obj in self.objects],
    #         "history": self.named_history(limit=1),  # only latest position
    #     }


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
