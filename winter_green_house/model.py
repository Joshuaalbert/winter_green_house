import dataclasses
from functools import partial
from typing import NamedTuple

import astropy.units as au
import jax
import jax.numpy as jnp
import pylab as plt
from jax import lax

from winter_green_house.astropy_utils import quantity_to_jnp
from winter_green_house.serialisable_base_model import SerialisableBaseModel


class SimulationResults(NamedTuple):
    t: jax.Array
    energy_dot: jax.Array
    energy: jax.Array
    temperature: jax.Array


def deg_C(t: float):
    return (t + 273.15) * au.K


def deg_K_to_C(t: float):
    return t - 273.15


class ThermalConductivity(SerialisableBaseModel):
    # Acrylic	0.20
    # Epoxy	0.17
    # Epoxy glass fibre	0.23
    # Nylon 6	0.25
    # Polyethylene, low density (PEL)	0.33
    # Polyethylene, high density (PEH)	0.50
    # PTFE	0.25
    # PVC	0.19
    acryllic: au.Quantity = 0.20 * au.W / au.K / au.m
    epoxy: au.Quantity = 0.17 * au.W / au.K / au.m
    epoxy_glass_fibre: au.Quantity = 0.23 * au.W / au.K / au.m
    nylon_6: au.Quantity = 0.25 * au.W / au.K / au.m
    polyethylene_low_density: au.Quantity = 0.33 * au.W / au.K / au.m
    polyethylene_high_density: au.Quantity = 0.50 * au.W / au.K / au.m
    ptfe: au.Quantity = 0.25 * au.W / au.K / au.m
    pvc: au.Quantity = 0.19 * au.W / au.K / au.m


@dataclasses.dataclass(eq=False)
class NestedHemiSpheres:
    radii: au.Quantity

    # Internal heat source
    source_power: au.Quantity

    rho_air: au.Quantity = 1.225 * au.kg / au.m ** 3  # density of air
    c_p_air: au.Quantity = 700 * au.J / au.kg / au.K  # specific heat capacity of air
    k_air: au.Quantity = 0.025 * au.W / au.K / au.m  # thermal conductivity of air

    # Interface parameters
    delta: au.Quantity = 0.1 * au.m  # 10cm, thickness of the interface
    k: au.Quantity = ThermalConductivity().pvc  # thermal conductance of interface

    # Outside Air

    T_min: au.Quantity = deg_C(-30.)  # Cold winter night
    T_max: au.Quantity = deg_C(-10.)  # Cold winter day
    t_max: au.Quantity = 15 * au.h  # Around 15:00

    # Boundary conditions
    T_g: au.Quantity = deg_C(4.)  # temperature of the ground, close to annual average
    delta_g: au.Quantity = 1 * au.m  # thickness of the ground
    k_g: au.Quantity = 0.3 * au.W / au.K / au.m  # thermal conductance of ground

    def __post_init__(self):
        # ensure units
        if not self.radii.unit.is_equivalent(au.m):
            raise ValueError(f"Expected units of length, got {self.radii.unit}")
        if not self.rho_air.unit.is_equivalent(au.kg / au.m ** 3):
            raise ValueError(f"Expected units of density, got {self.rho_air.unit}")
        if not self.c_p_air.unit.is_equivalent(au.J / au.kg / au.K):
            raise ValueError(f"Expected units of specific heat capacity, got {self.c_p_air.unit}")

        if not self.delta.unit.is_equivalent(au.m):
            raise ValueError(f"Expected units of length, got {self.delta.unit}")
        if not self.k.unit.is_equivalent(au.W / au.K / au.m):
            raise ValueError(f"Expected units of thermal conductance, got {self.k.unit}")

        if not self.T_g.unit.is_equivalent(au.K):
            raise ValueError(f"Expected units of temperature, got {self.T_g.unit}")
        if not self.delta_g.unit.is_equivalent(au.m):
            raise ValueError(f"Expected units of length, got {self.delta_g.unit}")
        if not self.k_g.unit.is_equivalent(au.W / au.K / au.m):
            raise ValueError(f"Expected units of thermal conductance, got {self.k_g.unit}")

        if not self.T_min.unit.is_equivalent(au.K):
            raise ValueError(f"Expected units of temperature, got {self.T_min.unit}")
        if not self.T_max.unit.is_equivalent(au.K):
            raise ValueError(f"Expected units of temperature, got {self.T_max.unit}")
        if not self.t_max.unit.is_equivalent(au.s):
            raise ValueError(f"Expected units of time, got {self.t_max.unit}")

        if not self.source_power.unit.is_equivalent(au.W):
            raise ValueError(f"Expected units of power, got {self.source_power.unit}")

        if len(self.source_power) != len(self.radii):
            raise ValueError(f"Expected source power for each shell, got {len(self.source_power)}")

        def _hemi_sphere_volume(r):
            return 2 / 3 * jnp.pi * r ** 3

        def _circle_area(r):
            return jnp.pi * r ** 2

        def _hemi_sphere_area(r):
            return 2 * jnp.pi * r ** 2

        self.alpha_air = self.k_air / self.rho_air / self.c_p_air # diffusivity

        volumes = [
            _hemi_sphere_volume(self.radii[0])
        ]  # V
        shell_areas = [
            _hemi_sphere_area(self.radii[0])
        ]  # A
        ground_areas = [
            _circle_area(self.radii[0])
        ]  # a
        for i in range(1, len(self.radii)):
            V_i = _hemi_sphere_volume(self.radii[i]) - volumes[i - 1]
            A_i = _hemi_sphere_area(self.radii[i])
            a_i = _circle_area(self.radii[i]) - ground_areas[i - 1]

            volumes.append(V_i)
            shell_areas.append(A_i)
            ground_areas.append(a_i)
        self.V = au.Quantity(volumes)
        self.A = au.Quantity(shell_areas)
        self.a = au.Quantity(ground_areas)

    def _outside_air_temp_jax(self, t: jax.Array) -> jax.Array:
        # As a function of time
        T_min = quantity_to_jnp(self.T_min)
        T_max = quantity_to_jnp(self.T_max)
        t_max = quantity_to_jnp(self.t_max)
        return T_min + 0.5 * (T_max - T_min) * (1 + jnp.cos(2. * jnp.pi / 86400. * (t - t_max)))

    def _internal_heat_source_jax(self, t: jax.Array) -> jax.Array:
        # As a function of time
        source_power = quantity_to_jnp(self.source_power)
        return source_power

    def _compute_temperature_from_energy(self, energy_i: jax.Array) -> jax.Array:
        # Use E = rho * V * c_p * T
        V = quantity_to_jnp(self.V)  # [n]
        rho = quantity_to_jnp(self.rho_air)
        c_p = quantity_to_jnp(self.c_p_air)
        temperature_i = energy_i / (rho * V * c_p)
        return temperature_i

    def _compute_energy_from_temperature(self, temperature_i: jax.Array) -> jax.Array:
        # Use E = rho * V * c_p * T
        V = quantity_to_jnp(self.V)
        rho = quantity_to_jnp(self.rho_air)
        c_p = quantity_to_jnp(self.c_p_air)
        energy_i = rho * V * c_p * temperature_i
        return energy_i

    def __repr__(self):
        s = f"NestedHemiSpheres(\n"
        for i in range(len(self.radii)):
            s += f"  Shell {i}:\n"
            s += f"    Radius: {self.radii[i]}\n"
            s += f"    Volume: {self.V[i]}\n"
            s += f"    Area: {self.A[i]}\n"
            s += f"    Ground Area: {self.a[i]}\n"
        return s

    def _compute_energy_dot_jax(self, temperature_i: jax.Array, t: jax.Array) -> jax.Array:
        n = len(self.radii)
        V = quantity_to_jnp(self.V)
        A = quantity_to_jnp(self.A)
        a = quantity_to_jnp(self.a)
        delta = quantity_to_jnp(self.delta)
        k = quantity_to_jnp(self.k, 'W/K/m')
        T_g = quantity_to_jnp(self.T_g)
        delta_g = quantity_to_jnp(self.delta_g)
        k_g = quantity_to_jnp(self.k_g, 'W/K/m')

        # Outside air temperature
        T_e = self._outside_air_temp_jax(t)
        # Internal heat source
        heating = self._internal_heat_source_jax(t)  # [n]

        # All volumes interface with next volume, and ground.
        # Outermost shell is exposed to environment.
        def _E_i_dot_due_to_heat_transfer_to_j(thermal_conductance, delta, T_i, T_j, A):
            q = thermal_conductance * (T_j - T_i) / delta
            return q * A

        energy_dot = []
        for i in range(V.shape[0]):
            # Transfer to ground
            energy_i_dot = _E_i_dot_due_to_heat_transfer_to_j(
                thermal_conductance=k_g, delta=delta_g, T_i=temperature_i[i], T_j=T_g, A=a[i]
            )

            # internal source
            energy_i_dot += heating[i]

            if i < n - 1:  # Loss due to transfer to next shell
                energy_i_dot += _E_i_dot_due_to_heat_transfer_to_j(
                    thermal_conductance=k, delta=delta, T_i=temperature_i[i], T_j=temperature_i[i + 1], A=A[i]
                )

            if i > 0:  # Gain due to transfer from previous shell
                energy_i_dot -= _E_i_dot_due_to_heat_transfer_to_j(
                    thermal_conductance=k, delta=delta, T_i=temperature_i[i - 1], T_j=temperature_i[i], A=A[i - 1]
                )

            if i == n - 1:
                # Transfer to outside environment
                energy_i_dot += _E_i_dot_due_to_heat_transfer_to_j(
                    thermal_conductance=k, delta=delta, T_i=temperature_i[i], T_j=T_e, A=A[i]
                )
            energy_dot.append(energy_i_dot)

        return jnp.asarray(energy_dot)

    @partial(jax.jit, static_argnames=['self', 'num_timesteps'])
    def _simulate_jax(self, start_time: jax.Array, duration: jax.Array, initial_temperature: jax.Array,
                      num_timesteps: int) -> SimulationResults:
        dt = duration / (num_timesteps - 1)

        def body_fn(carry, t):
            (energy, temperature) = carry
            energy_dot = self._compute_energy_dot_jax(
                temperature_i=self._compute_temperature_from_energy(energy),
                t=t
            )
            energy += energy_dot * dt
            temperature = self._compute_temperature_from_energy(energy)
            return (energy, temperature), (energy_dot, energy, temperature)

        times = start_time + jnp.linspace(0, duration, num_timesteps)
        init_carry = (self._compute_energy_from_temperature(initial_temperature), initial_temperature)
        _, (energy_dot, energy, temperature) = lax.scan(
            body_fn,
            init_carry,
            times
        )
        return SimulationResults(
            t=times,
            energy_dot=energy_dot,
            energy=energy,
            temperature=temperature
        )

    def simulate(self, start_time: au.Quantity,
                 duration: au.Quantity,
                 dt: au.Quantity,
                 initial_temperature: au.Quantity) -> SimulationResults:

        n = len(self.radii)
        # Ensure units
        if not start_time.unit.is_equivalent(au.s):
            raise ValueError(f"Expected units of time, got {start_time.unit}")
        if not duration.unit.is_equivalent(au.s):
            raise ValueError(f"Expected units of time, got {duration.unit}")
        if not dt.unit.is_equivalent(au.s):
            raise ValueError(f"Expected units of time, got {dt.unit}")
        if not initial_temperature.unit.is_equivalent(au.K):
            raise ValueError(f"Expected units of temperature, got {initial_temperature.unit}")

        if initial_temperature.shape != self.radii.shape:
            raise ValueError(f"Expected initial temperature shape {(n,)}, got {initial_temperature.shape}")

        num_simulate_steps = int(duration / dt)

        results = self._simulate_jax(
            start_time=quantity_to_jnp(start_time, 's'),
            initial_temperature=quantity_to_jnp(initial_temperature, 'K'),
            duration=quantity_to_jnp(duration, 's'),
            num_timesteps=num_simulate_steps
        )

        print(results)

        return results


def test_simulate():
    radii = [5, 6, 7, 8.] * au.m

    # Ground temperature
    no_heating_model = NestedHemiSpheres(
        radii=radii,
        source_power=[0, 0, 0, 0] * au.W
    )

    run_data = dict(
        start_time=12 * au.h,
        duration=24 * au.h,
        dt=0.001 * au.h,
        initial_temperature=au.Quantity([deg_C(15), deg_C(10), deg_C(-20), deg_C(-30)])
    )

    no_heating_results = no_heating_model.simulate(
        **run_data
    )

    heating_model = NestedHemiSpheres(
        radii=radii,
        source_power=[4000, 0, 0, 0] * au.W
    )
    heating_results = heating_model.simulate(
        **run_data
    )

    n = len(radii)

    outside_air_temperature = jax.vmap(heating_model._outside_air_temp_jax)(heating_results.t)

    # E-dot
    fig, axs = plt.subplots(n, 1, figsize=(10, 10), sharex=True, squeeze=False)
    for i in range(n):
        axs[i][0].plot(no_heating_results.t, no_heating_results.energy_dot[:, i], label="No Heating")
        axs[i][0].plot(heating_results.t, heating_results.energy_dot[:, i], label="Heating")
        axs[i][0].set_ylabel(f"Power (W/s) - Shell {i}")
        axs[i][0].legend()
    axs[-1][0].set_xlabel("Time (s)")
    plt.show()

    # E
    fig, axs = plt.subplots(n, 1, figsize=(10, 10), sharex=True, squeeze=False)
    for i in range(n):
        axs[i][0].plot(no_heating_results.t, no_heating_results.energy[:, i], label="No Heating")
        axs[i][0].plot(heating_results.t, heating_results.energy[:, i], label="Heating")
        axs[i][0].set_ylabel(f"Energy (J) - Shell {i}")
        axs[i][0].legend()
    axs[-1][0].set_xlabel("Time (s)")
    plt.show()

    # T
    fig, axs = plt.subplots(n, 1, figsize=(10, 10), sharex=True, squeeze=False)
    for i in range(n):
        axs[i][0].plot(no_heating_results.t, deg_K_to_C(no_heating_results.temperature[:, i]), label="No Heating")
        axs[i][0].plot(heating_results.t, deg_K_to_C(heating_results.temperature[:, i]), label="Heating")
        axs[i][0].set_ylabel(f"Temperature (C) - Shell {i}")
        if i == n - 1:
            axs[i][0].plot(heating_results.t, deg_K_to_C(outside_air_temperature), label="Outside Air Temp",
                           linestyle='--')
        axs[i][0].legend()
    axs[-1][0].set_xlabel("Time (s)")
    plt.show()
