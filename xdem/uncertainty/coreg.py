# Copyright (c) 2026 GeoUtils developers
#
# Adapted for xDEM from the GeoUtils uncertainty simulation module:
# https://github.com/glaciohack/geoutils
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tools to help manage coregistration uncertainty."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from typing import Any, Literal

import geoutils as gu
import numpy as np
import pandas as pd
from geoutils._misc import import_optional
from geoutils.uncertainty import PropagationSummary as GeoUtilsPropagationSummary
from geoutils.uncertainty.propagation import simulate
from pyproj import CRS

from xdem.coreg import Coreg
from xdem.coreg.affine import ICP, LZD, _coreg_error_inputs
from xdem.coreg.base import _as_eager_elevation, translations_rotations_from_matrix
from xdem.uncertainty.propagation import _select_outputs


@dataclass(frozen=True, kw_only=True)
class PropagationSummary(GeoUtilsPropagationSummary):  # type: ignore[misc]
    """
    Fitted coregistration parameters and their uncertainty, with optional Coregs from numerical simulations.

    The summary includes estimate, mean and std as pandas Series indexed by tx, ty, tz, rx, ry and rz.
    Translations use coordinate units, rotations are Euler angles in radians about the centroid in output.

    With return_coregs=True, coregs stores the fitted Coreg objects and their fit metadata for each simulation
    number.

    plot() displays translations and rotations on separate axes, with the original fit and mean +/- one
    standard deviation. For example::

        summary = xdem.uncertainty.propagate_uncertainty_coreg(
            reference, moving, xdem.coreg.LZD(), error_structure={"tba": errors}, return_coregs=True,
        )
        summary.plot()
        for number, fitted in summary.coregs.items():
            print(number, fitted.meta["outputs"])

    :param coregs: Optional fitted Coreg objects indexed by simulation number, starting at 1.
    """

    coregs: Mapping[int, Coreg] | None = None

    def plot(self, axes: Sequence[Any] | None = None, **kwargs: Any) -> tuple[Any, Any]:
        """
        Plot parameter means with one standard deviation and mark the original fitted values.

        :param axes: Optional Matplotlib axes for translations and rotations.
        :param kwargs: Additional arguments for Matplotlib errorbar().

        :returns: Translation and rotation axes.
        """

        import_optional("matplotlib")
        import matplotlib.pyplot as plt

        if axes is None:
            _, axes = plt.subplots(1, 2, figsize=(8, 3), constrained_layout=True)
        translation_axis, rotation_axis = axes
        units = (self.output.get("translation_units", "coordinate units"), "radian")
        kwargs = {"fmt": "o", "capsize": 3, "label": "Mean +/- 1 std", **kwargs}

        # Separate coordinate and angular units
        for axis, names, title, unit in zip(
            (translation_axis, rotation_axis),
            (("tx", "ty", "tz"), ("rx", "ry", "rz")),
            ("Translation", "Rotation"),
            units,
        ):
            parameters = list(names)
            axis.errorbar(parameters, self.mean.loc[parameters], yerr=self.std.loc[parameters], **kwargs)
            axis.plot(parameters, self.estimate.loc[parameters], "x", label="Original fit")
            axis.set_ylabel(f"{title} ({unit})")
            axis.legend()
        return translation_axis, rotation_axis


def propagate_uncertainty_coreg(
    reference_elev: Any,
    to_be_aligned_elev: Any,
    coreg_method: Coreg,
    *,
    error_structure: gu.ErrorStructure | Mapping[str, gu.ErrorStructure],
    error_predictors: Mapping[str, Mapping[str, Any]] | None = None,
    method: Literal["analytical", "numerical"] = "numerical",
    inlier_mask: Any = None,
    weights: Any = None,
    subsample: int | float | None = None,
    random_state: int | np.random.Generator | None = None,
    fit_kwargs: Mapping[str, Any] | None = None,
    return_coregs: bool = False,
    **kwargs: Any,
) -> PropagationSummary:
    """
    Propagate and summarize uncertainties on translations and rotations from coregistration.

    Numerical propagation adds source errors over many simulations, repeating fit() with a fixed sampling seed.
    Analytical propagation supports linearized LZD and point-to-plane ICP.

    :param reference_elev: Reference raster or point cloud.
    :param to_be_aligned_elev: Raster or point cloud to align.
    :param coreg_method: Coregistration method, copied before fitting.
    :param error_structure: Independent models keyed by "ref" and/or "tba". A single model describes tba.
    :param error_predictors: Magnitude predictors keyed by the same input names.
    :param method: Repeated numerical fits, LZD fit uncertainties, or analytical point-to-plane ICP uncertainty.
    :param inlier_mask: Stable observations passed to fit().
    :param weights: Observation weights passed to fit().
    :param subsample: Optional replacement for the method's sample size.
    :param random_state: Seed controlling error draws and the fixed fit sample.
    :param fit_kwargs: Additional arguments passed unchanged to fit().
    :param return_coregs: Store each successful numerical fit as a Coreg, including its fit metadata.
    :param kwargs: GeoUtils simulation options, including n_samples, at, return_samples and quantiles.
        random_field_options passes options to generate_random_field(), which approximates correlated fields by
        default. Pass {"gpytorch_inducing_points": None} for exact in-memory draws.

    :returns: A propagation summary with six labelled parameters and optional fitted Coregs.
    """

    # 1/ Check inputs
    if method not in {"analytical", "numerical"}:
        raise ValueError("method must be 'analytical' or 'numerical'.")
    if method == "analytical" and kwargs.get("return_samples", False):
        raise ValueError("Analytical propagation cannot return simulation samples.")
    if method == "analytical" and return_coregs:
        raise ValueError("Analytical propagation cannot return simulated Coreg objects.")
    models, predictor_inputs = _coreg_error_inputs(error_structure, error_predictors)
    if not models:
        raise TypeError("Supply at least one coregistration error model.")
    if "error_applied_to" in kwargs or "predictors" in kwargs:
        raise TypeError("Use error_structure and error_predictors mappings keyed by 'ref' and 'tba'.")

    # Check unsupported ICP estimators before reading elevations or running a fit
    if method == "analytical" and isinstance(coreg_method, ICP):
        coreg_method._check_analytical_uncertainty()

    # 2/ Prepare elevations and fitting options
    reference = _as_eager_elevation(reference_elev)
    aligned = _as_eager_elevation(to_be_aligned_elev)
    options = dict(fit_kwargs or {})
    options.update(inlier_mask=inlier_mask, weights=weights)
    if subsample is not None:
        options["subsample"] = subsample
    # Resolve terrain predictors separately on each uncertain DEM
    from xdem import terrain

    sources = {"ref": reference, "tba": aligned}
    resolved = {}
    for name, values in predictor_inputs.items():
        resolved[name] = {
            key: (
                terrain.get_terrain_attribute(sources[name], value)
                if isinstance(value, str)
                and isinstance(sources[name], gu.Raster)
                and value in terrain.available_attributes
                else value
            )
            for key, value in values.items()
        }
    legacy_errors = any(key in options for key in ("sig_ref", "sig_tba", "corr_ref", "corr_tba"))
    error_aware = isinstance(coreg_method, LZD)
    if isinstance(coreg_method, ICP):
        try:
            coreg_method._check_analytical_uncertainty()
        except NotImplementedError:
            pass
        else:
            error_aware = True
    if error_aware and not legacy_errors:
        options.update(error_structure=models, error_predictors=resolved)
    operation_seed = int(np.random.default_rng(random_state).integers(0, 2**32))

    def fit(ref: Any, tba: Any) -> Coreg:
        """We fit a copy using the same observation sample for every realization."""

        return coreg_method.copy().fit(ref, tba, random_state=operation_seed, **options)

    # 3/ Fit the supplied inputs and errors
    if method == "analytical" and isinstance(coreg_method, ICP):
        if reference.crs != aligned.crs:
            raise ValueError("Analytical ICP requires inputs in the same CRS.")
    original = fit(reference, aligned)

    # 4/ Extract transformation parameters and metadata
    centroid = getattr(original, "centroid", lambda: None)()
    center = np.zeros(3) if centroid is None else np.asarray(centroid)
    labels = pd.Index(["tx", "ty", "tz", "rx", "ry", "rz"])

    def parameters(fitted: Coreg) -> pd.Series:
        """Read matrix parameters, and correct to original centroid."""

        matrix = fitted.to_matrix().copy()
        fitted_centroid = getattr(fitted, "centroid", lambda: None)()
        fitted_center = np.zeros(3) if fitted_centroid is None else np.asarray(fitted_centroid)
        matrix[:3, 3] += (np.eye(3) - matrix[:3, :3]) @ (fitted_center - center)
        return pd.Series(
            translations_rotations_from_matrix(matrix, return_degrees=False),
            index=labels,
        )

    estimate = parameters(original)
    # Read coordinate units from either Rasterio or PyProj CRS objects
    crs = None if reference.crs is None else CRS.from_user_input(reference.crs)
    translation_units = crs.axis_info[0].unit_name if crs is not None and crs.is_projected else "coordinate units"
    output = {
        "centroid": centroid,
        "crs": reference.crs,
        "translation_units": translation_units,
        "rotation_units": "radian",
    }

    # 5/ Return analytical uncertainty
    if method == "analytical":
        if isinstance(original, ICP):
            covariance = original.meta["outputs"]["fitorbin"]["fit_pcov"]
            output["parameter_covariance"] = pd.DataFrame(covariance, index=labels, columns=labels)
            output["fixed_correspondences"] = True
        elif not isinstance(original, LZD) or "fit_perr" not in original.meta["outputs"].get("fitorbin", {}):
            raise ValueError("Analytical coregistration uncertainty requires an LZD fit with GLS/TLS uncertainties.")
        if isinstance(original, LZD) and "fit_pcov" in original.meta["outputs"]["fitorbin"]:
            native_covariance = original.meta["outputs"]["fitorbin"]["fit_pcov"]
            covariance = np.zeros((6, 6))
            covariance[: len(native_covariance), : len(native_covariance)] = native_covariance
            output["parameter_covariance"] = pd.DataFrame(covariance, index=labels, columns=labels)
        standard_errors = np.zeros(6)
        errors = original.meta["outputs"]["fitorbin"]["fit_perr"]
        standard_errors[: len(errors)] = errors
        return PropagationSummary(
            estimate=estimate,
            mean=estimate.copy(),
            std=pd.Series(standard_errors, index=labels),
            error_structure=error_structure,
            method="analytical",
            output=output,
            selection=_select_outputs(estimate, kwargs.pop("at", None)),
        )

    # 6/ Simulate numerical uncertainty
    fitted_coregs: list[Coreg] | None = [] if return_coregs else None

    def calculate(values: Mapping[str, Any]) -> pd.Series:
        """Refit the perturbed inputs and extract the six native parameters."""

        fitted = fit(values.get("ref", reference), values.get("tba", aligned))
        result = parameters(fitted)

        # Save fits that GeoUtils includes in its parameter summaries
        if fitted_coregs is not None and np.any(np.isfinite(result.to_numpy())):
            fitted_coregs.append(fitted)
        return result

    field_options = dict(kwargs.pop("random_field_options", {}) or {})

    def draw_error(generator: np.random.Generator) -> dict[str, Any]:
        """Draw independent native fields for the named elevation inputs."""

        values = {}
        for name, model in models.items():
            source = sources[name]
            error = model.generate_random_field(
                like=source, predictors=resolved.get(name), random_state=generator, **field_options
            )
            values[name] = source.copy(new_array=source.data + error.data)
        return values

    selection = _select_outputs(estimate, kwargs.pop("at", None))
    if (kwargs.get("return_samples", False) or kwargs.get("quantiles")) and selection.empty:
        raise ValueError("Select outputs with at when requesting samples or quantiles.")
    kwargs.setdefault("on_error", "warn")
    kwargs.setdefault("n_samples", 100)
    summary = simulate(
        estimate=estimate,
        draw_error=draw_error,
        calculate=calculate,
        error_structure=error_structure,
        selection=selection,
        random_state=random_state,
        output=output,
        **kwargs,
    )

    # Match saved fits to successful simulation numbers, including gaps after failed draws
    stored_coregs = None
    if fitted_coregs is not None:
        assert summary.n_samples is not None
        numbers = [number for number in range(1, summary.n_samples + 1) if number not in summary.failures]
        stored_coregs = dict(zip(numbers, fitted_coregs, strict=True))
    return PropagationSummary(
        **{item.name: getattr(summary, item.name) for item in fields(summary)}, coregs=stored_coregs
    )
