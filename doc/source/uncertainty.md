---
file_format: mystnb
mystnb:
  execution_timeout: 150
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: xdem-env
  language: python
  name: xdem
---
(uncertainty)=

# Uncertainty analysis

xDEM integrates uncertainty analysis tools from the recent literature that **rely on joint methods from two
scientific fields: spatial statistics and uncertainty quantification**.

While uncertainty analysis technically refers to both systematic and random errors, systematic errors of elevation data
are corrected using {ref}`coregistration` and {ref}`biascorr`, so we here refer to uncertainty analysis for **quantifying and
propagating random errors** (including structured errors).

In detail, xDEM provides tools to:

1. Estimate and model elevation **heteroscedasticity, i.e. variable random errors** (e.g., with terrain slope or stereo-correlation),
2. Estimate and model the **spatial correlation of random errors** (e.g., from sensor resolution or instrument noise),
3. Perform **error propagation to elevation derivatives** (e.g., a spatial average, slope or aspect).

These tools share an {class}`~xdem.ErrorStructure` with
[GeoUtils](https://geoutils.readthedocs.io/en/stable/uncertainty.html). xDEM adds **terrain predictors, coregistration
and propagation to elevation derivatives**; GeoUtils implements the underlying error models, spatial statistics
and random fields.

:::{admonition} More reading
:class: tip

For an introduction, see the **{ref}`spatial-stats`** and **{ref}`static-surfaces`** guide pages.
For variography, see [GeoUtils' spatial statistics](https://geoutils.readthedocs.io/en/stable/stats.html#stats-spatial)
and the [SciKit-GStat documentation](https://scikit-gstat.readthedocs.io/en/latest/).
:::

```{code-cell} ipython3
:tags: [remove-cell]

# Match the figure resolution and text size used by the other feature pages
from matplotlib import pyplot as plt

plt.rcParams["figure.dpi"] = 600
plt.rcParams["savefig.dpi"] = 600
plt.rcParams["font.size"] = 9
```

## Quick use

Elevation errors can be **estimated on static surfaces, then propagated to a derived quantity**.
Here, we use two coregistered DEMs near Longyearbyen and exclude glaciers from the estimation:

```{code-cell} ipython3
:tags: [hide-cell]
:mystnb:
:  code_prompt_show: "Show the code for opening example files and selecting stable terrain"
:  code_prompt_hide: "Hide the code for opening example files and selecting stable terrain"

import geoutils as gu
import numpy as np
import xdem
from matplotlib.ticker import FuncFormatter, MaxNLocator

# Open the reference and an already coregistered DEM
ref_dem = xdem.DEM(xdem.examples.get_path("longyearbyen_ref_dem"))
dem = xdem.DEM(xdem.examples.get_path("longyearbyen_tba_dem_coreg"))
glacier_outlines = gu.Vector(xdem.examples.get_path("longyearbyen_glacier_outlines")).crop(dem)
stable_terrain = ~glacier_outlines.create_mask(dem)
dh = ref_dem - dem
```

```{code-cell} ipython3
# Estimate errors from elevation differences on stable terrain
errors = dem.estimate_error_structure(
    ref_dem, mask=stable_terrain, precoreg=False, random_state=42,
    variogram_estimator="dowd", n_pairs=50_000, n_lags=18,
)

# Predict the elevation error from slope and maximum curvature, the default predictors
predictors = {"slope": dem.slope(), "max_curvature": dem.max_curvature()}
sig_dem = errors.predict_magnitude(predictors, like=dem)
```

```{code-cell} ipython3
:tags: [hide-input]
:mystnb:
:  code_prompt_show: "Show the code for plotting the figure"
:  code_prompt_hide: "Hide the code for plotting the figure"

# Show the differences used for estimation and the resulting error map
stable_dh = dh.copy()
stable_dh.set_mask(~stable_terrain)
fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
stable_dh.plot(ax=axes[0], cmap="RdBu", vmin=-10, vmax=10,
               cbar_title="Elevation differences (m)", title="Static surfaces")
sig_dem.plot(ax=axes[1], cmap="Purples", vmin=0, vmax=8,
             cbar_title=r"Elevation error (1$\sigma$, m)", title="Predicted error")
for axis in axes:
    glacier_outlines.plot(ref=dem, ax=axis, fc="none", ec="0.25", linewidth=0.5)
    axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value / 1000:g}"))
    axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value / 1000:g}"))
    axis.xaxis.set_major_locator(MaxNLocator(4))
    axis.set_xlabel("Easting (km)")
axes[0].set_ylabel("Northing (km)")
axes[1].set_yticklabels([])
plt.tight_layout()
plt.show()
```

We can then propagate these errors to a **mean elevation difference**, here over a 600 × 600 m area:

```{code-cell} ipython3
:tags: [hide-cell]
:mystnb:
:  code_prompt_show: "Show the code for selecting a small area"
:  code_prompt_hide: "Hide the code for selecting a small area"

# Select a 600 × 600 m area on the original 20 m grid
pixel_bounds = (500, 450, 530, 480)
detail_dem = dem.icrop(pixel_bounds)
detail_dh = dh.icrop(pixel_bounds)
detail_predictors = {name: values.crop(detail_dem) for name, values in predictors.items()}
```

```{code-cell} ipython3
# Propagate uncertainty to the mean over all finite pixels in the selected area
area_result = xdem.uncertainty.propagate_uncertainty(
    detail_dh, "spatial", error_structure=errors, predictors=detail_predictors,
    method="analytical",
    inputs={"spatial": {"areas": {"Selected area": None}}},
)
print(f"Uncertainty in the area mean: {area_result.std['Selected area']:.2f} m")
```

Here, `precoreg=False` uses the existing alignment; by default, paired elevations are first coregistered with
{ref}`LZD <lzd>`. Install `xdem[opt]` for the optional backends used on this page.

## Summary of available methods

Uncertainty analysis combines **spatial statistics to model errors** and **uncertainty propagation to quantify
their effect on derived quantities**. xDEM implements approaches for elevation data from
[Rolstad et al. (2009)](https://doi.org/10.3189/002214309789470950) and
[Hugonnet et al. (2022)](https://doi.org/10.1109/JSTARS.2022.3188922).
These extend simpler models, including those used by
[Anderson (2019)](https://doi.org/10.1002/esp.4551).

### Modelling the structure of error

Elevation errors often have **several correlation ranges**, from pixel-scale noise to broad patterns across a DEM.
Rolstad et al. (2009) account for these different scales; Hugonnet et al. (2022) also account for
**heteroscedasticity**, or error magnitudes that vary with terrain or sensor properties.

| Approach | Variable magnitude | Single correlation range | Several correlation ranges |
| --- | :---: | :---: | :---: |
| Basic | No | Yes | No |
| Rolstad et al. (2009) | No | Yes | Yes |
| Hugonnet et al. (2022), default | Yes | Yes | Yes |

The more general models also describe constant errors or a single correlation range when the data support them.
We use the **NMAD** for error magnitude and **Dowd's variogram estimator** in the example above to limit the
influence of outliers. See **{ref}`robust-estimators`** for background.

### Propagating errors to spatial averages

Exact propagation evaluates the covariance of **every pair of observations**, with quadratic computing cost.
Rolstad et al. (2009) and Hugonnet et al. (2022) provide faster approximations with different assumptions:

| Approach | Accuracy | Computing cost | Assumptions |
| --- | --- | --- | --- |
| Exact covariance sum | Exact for the fitted error model | Quadratic in the number of observations | Any area shape and spatially varying magnitudes |
| Rolstad et al. (2009) | Conservative circular approximation | Fast radial integration | Stationary errors over a nearly circular, contiguous area |
| Hugonnet et al. (2022) | Sampled approximation | Linear for a fixed pair sample | Nearly stationary variance for the effective sample approximation |

The **effective sample size** expresses how correlation reduces the information in a spatial average.
We detail its formulation and the current covariance calculations in {ref}`uncertainty-spatial-averages` and
{ref}`uncertainty-large-areas` below.

(error-struc)=
## Estimating the spatial structure of error

{func}`~xdem.uncertainty.estimate_error_structure`<br>
{class}`~xdem.ErrorStructure`

Elevation errors are described by **their magnitude and spatial correlation**. Estimation combines the three
steps below into an error structure that can be inspected and reused for propagation.
The same estimation is available for {class}`~xdem.DEM` and {ref}`elevation point clouds <epc-class>`.

The structure combines independent error components, each with a magnitude and an optional correlation model.
Their variances add; `errors.info()` summarizes the fitted components and ranges.

(spatialstats-heterosc)=
### Elevation differences as an error proxy

Elevation differences on **static surfaces act as an error proxy**. To assign their uncertainty to the target DEM,
choose `other_precision`:

- **`"negligible"` (default)** assumes that the reference error is small enough to ignore. The difference errors
  then approximate those of the target DEM, as in this example.
- **`"same"`** assumes independent errors with the same magnitude and spatial correlation in both DEMs.
  The differences are divided by $\sqrt{2}$ before estimation.

To model **elevation differences themselves**, call `dh.estimate_error_structure()` without a second DEM. In that
case, supply predictor arrays or rasters explicitly, because terrain attributes cannot be derived from a difference
map. See the **{ref}`static-surfaces` guide** and
[GeoUtils' estimation examples](https://geoutils.readthedocs.io/en/stable/uncertainty.html#uncertainty-estimation).

### Heteroscedasticity

**Heteroscedasticity** describes variability in random elevation errors.
By default, xDEM uses terrain slope and maximum curvature to explain this variability. Other predictors can include
stereo-correlation or interferometric coherence, supplied as `predictors={"name": values}`.

Magnitudes are estimated by grouping the error proxy into predictor bins and calculating its spread, using the
normalized median absolute deviation (**NMAD**) by default. A model interpolates between these estimates.
The stored diagnostics show both the error magnitude and the number of observations in each bin:

```{code-cell} ipython3
:tags: [hide-input]
:mystnb:
:  code_prompt_show: "Show the code for plotting the magnitude diagnostics"
:  code_prompt_hide: "Hide the code for plotting the magnitude diagnostics"

# Plot the variable component and the observations used to estimate it
fig, ax = plt.subplots(figsize=(7, 4))
fig.subplots_adjust(right=0.78)
magnitude_axes = errors.plot_magnitude(ax=ax)
magnitude_axes["statistic"].set(xlabel="Slope (°)", ylabel="Maximum curvature (100 / m)")
magnitude_axes["colorbar"].set_position((0.84, 0.11, 0.025, 0.5))
magnitude_axes["colorbar"].set_ylabel(r"Short-range error (1$\sigma$, m)")
magnitude_axes["count_y"].xaxis.set_major_locator(MaxNLocator(2))
plt.show()
```

The default model assigns this variable magnitude to a short-range component and adds a constant long-range
component. The error map above combines both. Use `predictors={}` to estimate constant magnitudes instead.
The map extends the fitted relationship beyond static surfaces, assuming that **the predictors also explain
errors on other terrain**, including glaciers.

For custom bins and magnitude models, see
[GeoUtils' error estimation](https://geoutils.readthedocs.io/en/stable/uncertainty.html#uncertainty-estimation).

### Spatial correlation of errors

Nearby elevation errors are often correlated. To estimate this dependence, the error proxy is **standardized by
its variable magnitude**, then pairs are grouped by separation distance into an empirical variogram.
A fitted model describes how correlation decreases with distance.

The default with terrain predictors fits **a Gaussian short-range component and a spherical long-range component**.

```{code-cell} ipython3
:tags: [hide-input]
:mystnb:
:  code_prompt_show: "Show the code for plotting the correlation diagnostics"
:  code_prompt_hide: "Hide the code for plotting the correlation diagnostics"

# Compare the standardized variogram and the fitted correlation over distance
fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
fig.subplots_adjust(left=0.1, bottom=0.16, top=0.78, right=0.98, wspace=0.4)
errors.plot_correlation(ax=axes[0])
fig.axes[-1].set_ylabel("Pairs")
axes[0].set(xlabel="Separation (m)", ylabel="Standardized semivariance")
distances = np.geomspace(1, 30_000, 300)
for name, component in errors.components.items():
    axes[1].plot(distances, component.predict_correlation(distances), "--", label=name.replace("_", " "))
axes[1].plot(distances, errors.predict_correlation(distances), color="0.2", label="Combined")
axes[1].set(xscale="log", xlabel="Separation (m)", ylabel="Error correlation", ylim=(-0.03, 1.03))
axes[1].legend(frameon=False)
plt.show()
```

`predict_correlation()` summarizes the combined correlation using typical component magnitudes. Propagation uses
the **local magnitudes at both observations in every pair**. The
[GeoUtils error model guide](https://geoutils.readthedocs.io/en/stable/uncertainty.html#uncertainty-model) explains
how magnitudes and correlations combine into covariance.

The longest-distance bins exceed the fitted model here; see {ref}`uncertainty-model-choices` before interpreting
the long-range component.

## Propagation of errors

{func}`~xdem.uncertainty.propagate_uncertainty`

Elevation errors affect **spatial averages, terrain attributes and coregistration parameters**.
Propagation uses the estimated error structure to quantify uncertainty in these results.
The returned summary stores the original result in `estimate`, its propagated mean in `mean`,
and its propagated standard deviation in `std`.

Use `method="analytical"` for the available formulas, or `method="numerical"` (default) to repeat the operation
with random error fields. Numerical examples and additional options are in {ref}`uncertainty-advanced` below.

(uncertainty-spatial-averages)=
### Spatial averages

**Correlated errors average out more slowly than independent errors.** For $N$ elevation differences, the
uncertainty in their mean depends on the covariance between every pair:

$$
\sigma_{\overline{dh}}^2 = \frac{1}{N^2}\sum_{i=1}^{N}\sum_{j=1}^{N}
\operatorname{Cov}(\epsilon_i, \epsilon_j).
$$

Here, $\epsilon_i$ is the error in observation $i$. Each covariance combines the error magnitudes at the two
observations and their correlation.
Following [Rolstad et al. (2009)](https://doi.org/10.3189/002214309789470950) and
[Hugonnet et al. (2022)](https://doi.org/10.1109/JSTARS.2022.3188922), we can express this through an
**effective sample size** $N_{eff}$:

$$
\sigma_{\overline{dh}} = \frac{\overline{\sigma_{dh}}}{\sqrt{N_{eff}}}.
$$

Here, $\overline{\sigma_{dh}}$ is the average local error magnitude. For constant, independent errors,
$N_{eff}=N$; positive spatial correlation reduces the effective sample size.
The example in **Quick use** accounts for this correlation, giving a larger uncertainty than assuming independent pixels:

```{code-cell} ipython3
:tags: [hide-input]
:mystnb:
:  code_prompt_show: "Show the code for plotting the figure"
:  code_prompt_hide: "Hide the code for plotting the figure"

# Compare the propagated uncertainty with an assumption of independent pixels
local_magnitudes = errors.predict_magnitude(detail_predictors, like=detail_dem).data
valid = ~np.ma.getmaskarray(detail_dh.data)
independent_std = np.sqrt(np.sum(local_magnitudes[valid] ** 2)) / np.count_nonzero(valid)
mean_dh = area_result.estimate["Selected area"]
fig, axes = plt.subplots(1, 2, figsize=(8, 3.3))
detail_dh.plot(ax=axes[0], cmap="RdBu", vmin=-10, vmax=10,
               cbar_title="Elevation difference (m)", title="Selected area")
axes[0].xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value / 1000:g}"))
axes[0].yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value / 1000:g}"))
axes[0].set(xlabel="Easting (km)", ylabel="Northing (km)")
axes[1].errorbar([0, 1], [mean_dh, mean_dh],
                 yerr=[independent_std, area_result.std["Selected area"]],
                 fmt="o", color="0.2", capsize=5)
axes[1].set(xticks=[0, 1], xticklabels=["Independent", "Correlated"],
            ylabel=r"Mean difference ± 1$\sigma$ (m)", xlim=(-0.6, 1.6))
axes[1].grid(axis="y", alpha=0.25)
plt.tight_layout()
plt.show()
```

This uses the assumption above that reference errors are negligible, so the DEM error structure also approximates
that of the elevation differences. If both DEMs contribute errors, estimate the structure of their differences
directly before propagating an elevation change.

Pass polygons or Boolean masks in `areas` to select other regions, and `statistic="integral"` in
`operation_options` for a volume change. The {ref}`uncertainty-large-areas` example below covers larger regions.

### Terrain attributes

Elevation errors also affect terrain derivatives. Analytical propagation is available for **slope and aspect**, using
the derivatives of the selected terrain method. For example, compute slope uncertainty on the same DEM detail:

```{code-cell} ipython3
slope_result = xdem.uncertainty.propagate_uncertainty(
    detail_dem,
    "terrain",
    error_structure=errors,
    predictors=detail_predictors,
    method="analytical",
    operation_options={"attribute": "slope", "surface_fit": "Florinsky"},
)
```

```{code-cell} ipython3
:tags: [hide-input]
:mystnb:
:  code_prompt_show: "Show the code for plotting the figure"
:  code_prompt_hide: "Hide the code for plotting the figure"

fig, axes = plt.subplots(1, 2, figsize=(8, 3.3))
slope_result.estimate.plot(ax=axes[0], cmap="Reds", cbar_title="Slope (°)", title="Slope")
slope_result.std.plot(ax=axes[1], cmap="Purples", cbar_title=r"Slope error (1$\sigma$, °)",
                      title="Propagated uncertainty")
for axis in axes:
    axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value / 1000:g}"))
    axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value / 1000:g}"))
    axis.xaxis.set_major_locator(MaxNLocator(4))
    axis.set_xlabel("Easting (km)")
axes[0].set_ylabel("Northing (km)")
axes[1].set_yticklabels([])
plt.tight_layout()
plt.show()
```

These are **first-order approximations**, appropriate when elevation errors are small enough for a local
linearization. Flat terrain has undefined slope/aspect derivatives and is masked in the analytical uncertainty.
Use numerical propagation to explore larger errors or other terrain attributes, as below.

### Coregistration parameters

{func}`~xdem.uncertainty.propagate_uncertainty_coreg`

Elevation errors also affect the fitted **translations and rotations**. We can account for errors in the moving
DEM (`tba`), the reference (`ref`), or both. Here, we propagate the moving DEM's errors through an ICP fit:

```python
summary = xdem.uncertainty.propagate_uncertainty_coreg(
    ref_dem, dem, xdem.coreg.ICP(subsample=5000),
    error_structure={"tba": errors}, error_predictors={"tba": predictors}, inlier_mask=stable_terrain,
    method="analytical", random_state=42,
)
summary.std  # tx, ty, tz in coordinate units, rx, ry, rz in radians
summary.output["parameter_covariance"]  # Labelled 6 x 6 covariance
```

The helper fits a copy of the method and returns parameter standard deviations and covariance.
Analytical propagation assumes a **converged fit, fixed point pairs and a linear loss**. Use numerical
propagation when errors can change the point pairs, or with robust losses and trimming.
See {ref}`icp` and {ref}`lzd` for GLS/TLS fitting options.

(uncertainty-advanced)=
## Advanced use

(uncertainty-model-choices)=
### Customizing the estimated structure

The default paired-DEM model includes variable magnitudes and two correlation ranges. Simpler structures can be
specified explicitly. Here, we compare it with a constant magnitude and no spatial correlation:

```{code-cell} ipython3
constant_errors = dem.estimate_error_structure(
    ref_dem,
    mask=stable_terrain,
    precoreg=False,
    other_precision="negligible",
    predictors={},
    components={"measurement": {"magnitude": "constant", "correlation": None}},
    random_state=42,
)
```

```{code-cell} ipython3
:tags: [hide-input]
:mystnb:
:  code_prompt_show: "Show the code for plotting the figure"
:  code_prompt_hide: "Hide the code for plotting the figure"

# Hold curvature fixed to show how predicted error varies with slope
slope_values = np.linspace(0, 55, 100)
variable_magnitude = errors.predict_magnitude(
    {"slope": slope_values, "max_curvature": np.zeros_like(slope_values)}
)
fig, ax = plt.subplots(figsize=(6, 3), layout="constrained")
ax.plot(slope_values, variable_magnitude, label="Variable magnitude (curvature = 0)")
ax.axhline(constant_errors.predict_magnitude(), color="0.3", linestyle="--", label="Constant magnitude")
ax.set(xlabel="Slope (°)", ylabel=r"Elevation error (1$\sigma$, m)")
ax.legend(frameon=False)
plt.show()
```

The simpler structure discards both terrain dependence and correlation. Model choices should be checked against
the empirical diagnostics. In the real-data variogram above, the longest-distance bins continue to rise beyond
the fitted sill. Residual systematic trends or a scene too small to resolve the longest range can cause this.
Check the residual maps and the sensitivity to `max_lag` before relying on a fitted long-range component.
Use `errors.refit()` to change correlation models without resampling pairs, or construct
{class}`~xdem.ErrorComponent` objects from known magnitudes and models. See
[GeoUtils' model examples](https://geoutils.readthedocs.io/en/stable/uncertainty.html#uncertainty-model).

### Numerical propagation

Numerical propagation draws **spatially correlated error fields**, adds each to the elevations, repeats the operation,
and summarizes the results. `n_simulations` controls the number of repeats.

For a controlled comparison, we define a **1 m error magnitude and a Gaussian correlation with a 150 m effective
range**. This separates the propagation calculation from uncertainty in fitting the error model:

```{code-cell} ipython3
# Define a known error structure using the same classes as GeoUtils
simulation_errors = xdem.ErrorStructure([
    xdem.ErrorComponent(
        "measurement",
        magnitude=1.0,
        correlation=gu.Variogram.from_model("gaussian", effective_range=150, partial_sill=1),
    )
])

# Retain the simulated slope values at one pixel, while summarizing uncertainty everywhere
numerical_slope = xdem.uncertainty.propagate_uncertainty(
    detail_dem,
    "terrain",
    error_structure=simulation_errors,
    method="numerical",
    operation_options={"attribute": "slope", "surface_fit": "Florinsky"},
    random_field_options={"backend": "gstools"},
    n_simulations=200,
    random_state=42,
    at=[(15, 15)],
    return_samples=True,
)
```

```{code-cell} ipython3
:tags: [hide-input]
:mystnb:
:  code_prompt_show: "Show the code for plotting the figure"
:  code_prompt_hide: "Hide the code for plotting the figure"

# Compare with analytical propagation of the same known errors
from scipy.stats import norm

analytical_slope = xdem.uncertainty.propagate_uncertainty(
    detail_dem, "terrain", error_structure=simulation_errors, method="analytical",
    operation_options={"attribute": "slope", "surface_fit": "Florinsky"},
)
fig, axes = plt.subplots(1, 2, figsize=(8, 3.3))
analytical_std = analytical_slope.std.data.filled(np.nan)
numerical_std = numerical_slope.std.data.filled(np.nan)
common = np.isfinite(analytical_std) & np.isfinite(numerical_std)
xvalues = analytical_std[common]
yvalues = numerical_std[common]
axes[0].scatter(xvalues, yvalues, s=6, alpha=0.4)
limit = max(np.max(xvalues), np.max(yvalues)) * 1.05
axes[0].plot([0, limit], [0, limit], color="0.3", linestyle="--")
axes[0].set(xlabel="Analytical slope error (°)", ylabel="Numerical slope error (°)",
            xlim=(0, limit), ylim=(0, limit))
samples = np.asarray(numerical_slope.samples).ravel()
axes[1].hist(samples, bins=16, density=True, color="0.65", label="Numerical draws")
center = float(analytical_slope.estimate.data[15, 15])
spread = float(analytical_slope.std.data[15, 15])
slope_axis = np.linspace(center - 4 * spread, center + 4 * spread, 200)
axes[1].plot(slope_axis, norm.pdf(slope_axis, center, spread), color="0.2", label="First-order normal")
axes[1].set(xlabel="Slope at selected pixel (°)", ylabel="Density")
axes[1].legend(frameon=False)
plt.tight_layout()
plt.show()
```

The analytical and numerical spreads are close here, where errors are small relative to the terrain gradient.
Spectral random field backends approximate covariance; check their agreement at the spatial scales relevant to your result.
`estimate` always stores the result from the original DEM. `mean` can differ for nonlinear operations.
`at` selects the outputs whose draws are saved; it does not restrict propagation. Request `quantiles=(0.025, 0.975)`
for empirical intervals at those outputs. For custom operations, pass a callable that receives the perturbed DEM.

Generate fields directly with `simulation_errors.generate_random_field(like=detail_dem, backend="gstools")`.
[GeoUtils' simulations](https://geoutils.readthedocs.io/en/stable/uncertainty.html#uncertainty-simulation) explain the
backends and approximations, and its
[propagation guide](https://geoutils.readthedocs.io/en/stable/uncertainty.html#uncertainty-propagation) covers
interpolation, resampling and other spatial operations.

(uncertainty-large-areas)=
### Larger areas and effective samples

{func}`~xdem.uncertainty.spatial_error_propagation`<br>
{func}`~xdem.uncertainty.number_effective_samples`

Large areas need **an approximation to avoid evaluating every pair of observations**.
Following the pair sampling approach of
[Hugonnet et al. (2022)](https://doi.org/10.1109/JSTARS.2022.3188922), we sample the covariance sum.
The current calculation uses local error magnitudes at both observations, allowing spatially varying variance.
`subsample` sets how many observations are used as the second member of each pair; `None` evaluates all pairs.

For example, estimate uncertainty in the mean elevation change of Brombreen:

```{code-cell} ipython3
# Select Brombreen and propagate its mean elevation error without constructing a full covariance matrix
outline_brom = gu.Vector(glacier_outlines.gdf[glacier_outlines.gdf["NAME"] == "Brombreen"])
sig_dh_brom = xdem.uncertainty.spatial_error_propagation(
    [outline_brom], errors, support=dh, predictors=predictors, subsample=1000, random_state=42,
)[0]
mask_brom = outline_brom.create_mask(dh)
dh_brom = np.nanmean(dh[mask_brom])
neff = xdem.uncertainty.number_effective_samples(
    outline_brom, errors, support=dh, predictors=predictors, subsample=1000, random_state=42,
)
```

```{code-cell} ipython3
:tags: [hide-input]
:mystnb:
:  code_prompt_show: "Show the code for plotting the figure"
:  code_prompt_hide: "Hide the code for plotting the figure"

# Display the glacier and report the uncertainty of its mean elevation change
glacier_dh = dh.crop(outline_brom)
fig, ax = plt.subplots(figsize=(5, 4))
glacier_dh.plot(ax=ax, cmap="RdYlBu", vmin=-30, vmax=30, cbar_title="Elevation difference (m)")
outline_brom.plot(ref=glacier_dh, ax=ax, fc="none", ec="black", linewidth=1)
ax.set_title(f"Brombreen: {dh_brom:.2f} ± {sig_dh_brom:.2f} m\nEffective samples: {neff:.1f}")
ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value / 1000:g}"))
ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value / 1000:g}"))
ax.set(xlabel="Easting (km)", ylabel="Northing (km)")
plt.tight_layout()
plt.show()
```

A numeric area, rather than a spatial mask or polygon, uses the circular-area approximation of
[Rolstad et al. (2009)](https://doi.org/10.3189/002214309789470950). This is fast but assumes a stationary error
structure and a nearly circular, contiguous area. The sampled covariance calculation above instead uses the actual
observation locations, area shape and variable magnitudes. Increase the pair sample and check convergence when
precision in the propagated uncertainty matters.

```{note}
The former `DEM.estimate_uncertainty()` and `xdem.spatialstats` interfaces are deprecated. Use
`estimate_error_structure()` and its returned model for estimation, then `propagate_uncertainty()` or the spatial
helpers above for propagation. Generic binning, variography and error models now live in GeoUtils.
```
