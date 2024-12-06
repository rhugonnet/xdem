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
(cheatsheet)=

# Pair with reference data

Most analysis of **xDEM relies on independent, high-precision elevation data to use as reference**, whether for
coregistration, bias-corrections or uncertainty analysis.

[SlideRule](https://slideruleearth.io/) provides the ideal way to retrieve such data by accessing big data archives
of high-precision elevations, such as ICESat-2 or GEDI, efficiently in the cloud.

Below, a short example to coregister our example DEM with ICESat-2 ATL06 reference data.

```{code-cell} ipython3
:tags: [remove-cell]

# To get a good resolution for displayed figures
from matplotlib import pyplot
pyplot.rcParams['figure.dpi'] = 600
pyplot.rcParams['savefig.dpi'] = 600
pyplot.rcParams['font.size'] = 9  # Default 10 is a bit too big for coregistration plots
```

## Retrieving reference data

We load our example DEM data, and define the region of interest from its footprint.

```{code-cell} ipython3
from sliderule import sliderule, icesat2
import geoutils as gu
import xdem
import numpy as np

# Open example DEM
dem = xdem.DEM(xdem.examples.get_path("longyearbyen_ref_dem"))
dem.set_vcrs("EGM96")
dem = dem.to_vcrs("Ellipsoid")

# Derive inlier mask as glaciers
glacier_outlines = gu.Vector(xdem.examples.get_path("longyearbyen_glacier_outlines"))
inlier_mask = ~glacier_outlines.create_mask(dem)

# Define region of interest as DEM footprint
bounds = list(dem.get_bounds_projected(4326))
region = sliderule.toregion(bounds)["poly"]
```

We can then initialize the SlideRule client, and fetch the ICESat-2 ATL06 reference data in the region of interest
using {func}`icesat2.atl06sp`.

```{code-cell} ipython3
# Initialize SlideRule client
sliderule.init("slideruleearth.io")

# Define desired parameters for ICESat-2 ATL06
params = {
    "poly": region,
    "srt": icesat2.SRT_LAND,  # Surface-type
    "cnf": icesat2.CNF_SURFACE_HIGH,  # Confidence level
    "ats": 20.0,  # Minimum along-track spread
    "cnt": 10,  # Minimum count
}

# Request ATL06 subsetting in parallel
gdf = icesat2.atl06sp(params)
gdf = gdf[np.isfinite(gdf["h_li"])] # Keep valid data
gdf = gdf[gdf["atl06_quality_summary"]==0]  # Keep very high-confidence data
```

## Coregistration

Running the coregistration is as simple as passing the geodataframe to {func}`~xdem.coreg.Coreg.fit_and_apply`,
specifying the name of the column to use.

```{code-cell} ipython3
# Run a translation coregistration
nk = xdem.coreg.NuthKaab()
aligned_dem = nk.fit_and_apply(reference_elev=gdf, to_be_aligned_elev=dem, inlier_mask=inlier_mask, z_name="h_li", random_state=42)

# Print the estimated translation parameters
print([k+f': {nk.meta["outputs"]["affine"][k]:.2f} meters' for k in ["shift_x", "shift_y", "shift_z"]])
```

Let's visualize the improvement in elevation differences after coregistration.

```{code-cell} ipython3
:tags: [hide-input]
:mystnb:
:  code_prompt_show: "Show plotting code"
:  code_prompt_hide: "Hide plotting code"

# Derive hillshade for background
hs = dem.hillshade()
# Convert to same CRS
gdf = gdf.to_crs(dem.crs)
vect = gu.Vector(gdf)
# Mask areas not in inlier mask (glaciers here)
pc_mask = inlier_mask.astype("uint8").interp_points((gdf.geometry.x.values, gdf.geometry.y.values), method="nearest")
vect.ds = vect.ds[pc_mask == 1]
# Get point differences before and after
z_pc = dem.interp_points((vect.ds.geometry.x.values, vect.ds.geometry.y.values))
z_pc_aligned = aligned_dem.interp_points((vect.ds.geometry.x.values, vect.ds.geometry.y.values))
import warnings
# GeoPandas raising unexpected warning
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    vect["dh_bef"] = vect["h_li"] - z_pc
    vect["dh_aft"] = vect["h_li"] - z_pc_aligned

# Plot before and after
import matplotlib.pyplot as plt
f, ax = plt.subplots(1, 2)
ax[0].set_title("Before\ncoregistration")
hs.plot(cmap="Greys_r", add_cbar=False)
vect.plot(column="dh_bef", cmap='RdYlBu', vmin=-10, vmax=10, ax=ax[0], markersize=0.5, cbar_title="Elevation differences (m)")
ax[1].set_title("After\ncoregistration")
hs.plot(cmap="Greys_r", add_cbar=False)
vect.plot(column="dh_aft", cmap='RdYlBu', vmin=-10, vmax=10, ax=ax[1], markersize=0.5, legend=True, cbar_title="Elevation differences (m)")
_ = ax[1].set_yticklabels([])
plt.tight_layout()
```

We can print a coregistration summary using {func}`~xdem.coreg.Coreg.info`.

```{code-cell} ipython3
# Show full coregistration information
nk.info()
```