# Thermal Simulation and IR Rendering

## Bulk thermal model

Every registered body has temperature `T` in kelvin and thermal mass

```text
C = m c_p
```

where `m` is mass in kilograms and `c_p` is specific heat in J/(kg K). The
temperature update integrates the net heat rate:

```text
dT/dt = (Q_solar + Q_convection + Q_radiation
         + Q_contact + Q_internal) / C
```

Implemented heat-transfer terms include:

- Solar absorption from irradiance, absorptivity, projected area, and shadowing
- Forced/natural convection from air temperature, wind, and geometry
- Longwave exchange with the configured sky temperature using emissivity and
  the Stefan-Boltzmann relationship
- Contact conduction between registered PyBullet bodies
- Optional internal heat generation

Geometry comes from explicit dimensions or PyBullet AABBs. Mass comes from
PyBullet dynamics when available, otherwise density times volume. Contact
exchange snapshots temperatures before updates so results do not depend on
dictionary iteration order.

## Agent thermal inheritance

`Agent(ThermalBody, RenderableObject)` provides one consistent lifecycle:

- `attach_thermal()` registers a body after model and identity configuration.
- `temperature` exposes its primary thermal-object temperature.
- `sync_thermal_position()` aligns bodies without a PyBullet pose.
- `detach_thermal()` unregisters every body/link owned by the agent.

## Apparent detector temperature

The IR camera uses a broadband radiance approximation. For emissivity `epsilon`
and path transmission `tau`, detector radiance is approximated by

```text
L_detector = tau [epsilon T_surface^4
                  + (1 - epsilon) T_reflected^4]
             + (1 - tau) T_atmosphere^4
T_apparent = L_detector^(1/4)
```

This preserves the nonlinear radiometric relationship while avoiding an
expensive wavelength-resolved integration.

## GPU surface visualization

The physical solver supplies one bulk temperature per object. A camera-specific
GPU shader calculates each visible fragment using:

- Surface normal and solar incidence
- Low-frequency procedural material variation
- Texture luminance as a solar-absorption proxy, never as visible RGB color
- Object emissivity and surface-temperature baseline
- Reflected and atmospheric radiance
- Distance-dependent atmospheric extinction
- NETD-scaled temporal and fixed-pattern detector noise

Because per-pixel work runs in one GPU pass, Python only refreshes object-level
shader inputs when temperature changes exceed the detector NETD threshold.

## Automatic gain control and legend

The detector measurement range and displayed range are separate. With AGC
enabled, foreground temperatures determine a display window with a configurable
minimum span. This exposes differences of a few kelvin that would disappear
inside the full manufacturer-rated range.

The side legend uses the exact selected palette and active AGC range. Its lower,
middle, and upper labels show kelvin and Celsius estimates. It is visible only
for the IR camera and is included in composited `Z` screenshots.

## Validation and interpretation

`thermal_sanity_check.py` performs deterministic hot/cold conduction tests, a
0-30 m/s wind sweep, sun-versus-shade radiation tests, and a 16-case combined
matrix. It exports raw time-series/summary CSV files, five plots, and ten
machine-readable pass/fail checks. Analytical first-order conduction and
convection solutions provide numerical error bounds, while energy accounting
checks that integrated heat agrees with the simulated temperature change.

The included checker does not calculate a statistical p-value. If its
autocorrelated time samples are later used in an exploratory ANOVA-style
sensitivity analysis, the null hypothesis is equal mean response across tested
levels and the alternative is that at least one level differs. Such results are
sensitivity indicators, not formal inference from independent experiments.

## Limitations

- One bulk solved temperature per object, not a finite-element surface field
- Broadband `T^4` radiometry, not wavelength-resolved calibration
- Procedural surface gradients approximate material heterogeneity
- No calibrated lens distortion, blooming, dead pixels, or nonuniformity
  correction sequence
- Atmospheric extinction is a compact distance model rather than MODTRAN
