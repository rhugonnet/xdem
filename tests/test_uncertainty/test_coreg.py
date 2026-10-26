"""Tests for coregistration parameter uncertainty and propagation summaries."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import geoutils as gu
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from affine import Affine
from geoutils.stats.variography import VariogramModel
from scipy.stats import chi2

import xdem


@pytest.fixture
def icp_dem() -> gu.Raster:
    """Create a small curved DEM to work on coreg with rotation/translation."""

    rows, columns = np.indices((10, 12))
    x, y = (columns - 5.5) * 2, (4.5 - rows) * 3
    values = 40 + 5 * np.sin(x / 6) + 3 * np.cos(y / 8) + 2 * np.sin((x + y) / 7)
    return gu.Raster.from_array(
        values, Affine(2, 0, 500_000, 0, -3, 5_100_000), 32632, nodata=np.nan, area_or_point="Point"
    )


def _compare_simulated_uncertainty(
    reference: gu.Raster | gu.PointCloud,
    moving: gu.Raster | gu.PointCloud,
    method: xdem.coreg.AffineCoreg,
    options: dict[str, Any],
    only_translation: bool,
    *,
    n_samples: int = 20,
    check_mean: bool = False,
) -> xdem.uncertainty.PropagationSummary:
    """Compare analytical parameter uncertainties with repeated fits under the same source errors."""

    # Propagate identical errors and fitting options analytically and numerically
    analytical = xdem.uncertainty.propagate_uncertainty_coreg(reference, moving, method, method="analytical", **options)
    numerical = xdem.uncertainty.propagate_uncertainty_coreg(
        reference,
        moving,
        method,
        n_samples=n_samples,
        on_error="raise",
        random_field_options={"backend": "gpytorch", "gpytorch_inducing_points": None},
        **options,
    )

    # We use a chi-square interval to quantitatively constrain the allowed variation from number of simulations,
    # with an additional 0.1% error across parameters
    count = 3 if only_translation else 6
    probability = 0.001 / (2 * count)
    bounds = np.sqrt(chi2.ppf([probability, 1 - probability], n_samples - 1) / (n_samples - 1))
    ratios = numerical.std.iloc[:count] / analytical.std.iloc[:count]
    assert np.all((bounds[0] <= ratios) & (ratios <= bounds[1])), ratios

    # Check that every draw contributes
    assert numerical.n_success == n_samples
    assert numerical.n_valid.eq(n_samples).all()
    if only_translation:
        np.testing.assert_array_equal(numerical.std.iloc[3:], 0)

    # Check simulated means against the original fit within 4.5 standard errors
    if check_mean:
        mean_error = (numerical.mean.iloc[:count] - analytical.estimate.iloc[:count]) / analytical.std.iloc[:count]
        np.testing.assert_allclose(mean_error, 0, rtol=0, atol=4.5 / np.sqrt(n_samples))
    return analytical


class TestCoregistrationSummary:
    """Test module for plots of translation and rotation uncertainty."""

    @pytest.mark.parametrize("supplied_axes", [False, True])
    def test_plot__parameters(self, supplied_axes: bool) -> None:
        """Checks that parameter plots show simulated means, standard deviations and the original estimate."""

        import matplotlib.pyplot as plt

        # Create known translations and rotations with nonzero uncertainty in every parameter
        labels = pd.Index(["tx", "ty", "tz", "rx", "ry", "rz"])
        estimate = pd.Series([1, 2, 3, 0.1, 0.2, 0.3], index=labels)
        expected_mean = estimate + pd.Series([0.1, -0.2, 0.3, -0.01, 0.02, -0.03], index=labels)
        expected_std = pd.Series([0.2, 0.3, 0.4, 0.02, 0.03, 0.04], index=labels)
        selection = pd.DataFrame({"flat_index": np.arange(6), "estimate": estimate}, index=labels)
        summary = xdem.uncertainty.PropagationSummary(
            estimate=estimate,
            mean=expected_mean,
            std=expected_std,
            error_structure=gu.ErrorStructure([gu.ErrorComponent("measurement", 0.2)]),
            method="numerical",
            selection=selection,
            output={"translation_units": "metre"},
        )

        # Create or reuse plot axes
        axes = tuple(plt.subplots(1, 2)[1]) if supplied_axes else None
        plotted_axes = summary.plot(axes=axes)
        if axes is not None:
            assert plotted_axes == axes

        # Check means, error bars and the original fit
        for axis, names, label in zip(
            plotted_axes, (("tx", "ty", "tz"), ("rx", "ry", "rz")), ("Translation (metre)", "Rotation (radian)")
        ):
            mean = summary.mean.loc[list(names)].to_numpy()
            std = summary.std.loc[list(names)].to_numpy()
            endpoints = np.asarray(axis.collections[0].get_segments())[:, :, 1]
            np.testing.assert_allclose(axis.lines[0].get_ydata(orig=False), mean)
            np.testing.assert_allclose(endpoints, np.column_stack((mean - std, mean + std)))
            np.testing.assert_allclose(axis.lines[-1].get_ydata(orig=False), summary.estimate.loc[list(names)])
            assert axis.get_ylabel() == label
        plt.close(plotted_axes[0].figure)


class TestCoregistrationPropagation:
    """Test module for simulated parameters, saved fits and analytical propagation summaries."""

    @pytest.mark.parametrize(
        "return_coregs, return_samples, at",
        [
            (False, False, ["tz"]),
            (False, True, ["tz"]),
            (True, False, ["tz"]),
            (True, True, ["tz"]),
            (True, True, None),
        ],
    )
    def test_propagate_uncertainty_coreg__saved_coregs(
        self, return_coregs: bool, return_samples: bool, at: list[str] | None
    ) -> None:
        """Checks that saved Coregs match the parameter summaries and store each fit's metadata."""

        # 1/ Simulate vertical shifts
        # Create a vertical offset
        reference = gu.Raster.from_array(np.arange(48, dtype=float).reshape(6, 8), Affine(2, 0, 0, 0, -2, 12), 32632)
        source = reference + 5
        method = xdem.coreg.VerticalShift(vshift_reduc_func=np.mean)
        errors = gu.ErrorStructure([gu.ErrorComponent("measurement", 0.2)])

        # Select Coreg and sample outputs independently
        summary = xdem.uncertainty.propagate_uncertainty_coreg(
            reference,
            source,
            method,
            error_structure=errors,
            n_samples=4,
            random_state=2,
            return_coregs=return_coregs,
            return_samples=return_samples,
            at=at,
        )

        # 2/ Check saved fits and summaries
        # Check six parameter summaries and selected samples
        labels = ["tx", "ty", "tz", "rx", "ry", "rz"]
        assert type(summary) is xdem.uncertainty.PropagationSummary
        for quantity in (summary.estimate, summary.mean, summary.std, summary.n_valid):
            assert isinstance(quantity, pd.Series)
            assert quantity.index.tolist() == labels
        assert summary.estimate["tz"] == pytest.approx(-5)

        # Check successful draws, metadata and selected sample columns
        assert summary.n_success == 4
        assert summary.n_valid.eq(4).all()
        assert summary.output["crs"] == reference.crs
        assert "centroid" in summary.output
        if return_samples:
            assert summary.samples is not None
            assert summary.samples.shape == (4, len(labels if at is None else at))
            assert summary.samples.columns.tolist() == (labels if at is None else at)

        # Check optional outputs and the unchanged method
        assert (summary.samples is not None) == return_samples
        assert not method._fit_called
        if not return_coregs:
            assert summary.coregs is None
            return

        # Check separate Coregs and their vertical shifts
        assert summary.coregs is not None
        assert list(summary.coregs) == [1, 2, 3, 4]
        assert len({id(fitted) for fitted in summary.coregs.values()}) == 4
        parameters = pd.DataFrame(0.0, index=list(summary.coregs), columns=summary.estimate.index)
        for number, fitted in summary.coregs.items():
            assert fitted is not method
            assert fitted._fit_called
            shift = fitted.meta["outputs"]["affine"]["shift_z"]
            assert fitted.to_matrix()[2, 3] == shift
            parameters.loc[number, "tz"] = shift

        # Compare means, uncertainties and samples with saved fits
        np.testing.assert_allclose(parameters.mean(), summary.mean)
        np.testing.assert_allclose(parameters.std(), summary.std)
        if summary.samples is not None:
            np.testing.assert_array_equal(parameters.loc[:, summary.samples.columns], summary.samples)

    @pytest.mark.parametrize("only_translation", [True, False])
    @pytest.mark.parametrize(
        "solver, optimizer, legacy, seed",
        [
            pytest.param(xdem.coreg.LZD, "gls", False, 12, id="lzd-native"),
            pytest.param(xdem.coreg.LZD, "gls", True, 2, id="lzd-legacy-gls"),
            pytest.param(xdem.coreg.LZD, "tls", True, 2, id="lzd-legacy-tls"),
            pytest.param(xdem.coreg.ICP, "gls", False, 12, id="icp-integer-seed"),
            pytest.param(xdem.coreg.ICP, "gls", False, None, id="icp-no-seed"),
            pytest.param(xdem.coreg.ICP, "gls", False, np.random.default_rng(12), id="icp-generator"),
        ],
    )
    def test_propagate_uncertainty_coreg__stored_fit(
        self,
        icp_dem: gu.Raster,
        only_translation: bool,
        solver: type[xdem.coreg.AffineCoreg],
        optimizer: str,
        legacy: bool,
        seed: int | np.random.Generator | None,
    ) -> None:
        """Checks that analytical propagation refits supplied inputs and errors when the method is already fitted."""

        # 1/ Fit curved terrain using native models or the legacy LZD error arguments
        reference = icp_dem
        source = reference + 0.2
        errors = gu.ErrorStructure([gu.ErrorComponent("measurement", 0.1)])
        options: dict[str, Any] = {"force_opti": optimizer}
        if legacy:
            sigma = np.full(reference.shape, 0.1)
            correlation = (lambda distance: np.exp(-((distance / 2) ** 2)), 2)
            options.update(sig_tba=sigma, corr_tba=correlation)
            if optimizer == "tls":
                options.update(sig_ref=sigma, corr_ref=correlation)
        else:
            options["error_structure"] = {"tba": errors}
        method = solver(
            only_translation=only_translation, subsample=60, tolerance_translation=1e-6, tolerance_rotation=1e-6
        )
        fitted = method.fit(reference, source, random_state=seed, **options)
        matrix = fitted.to_matrix().copy()

        # 2/ Change both the elevations and errors, using the same seed for a fresh fit and the saved method
        changed_source = source + 0.2
        changed_errors = gu.ErrorStructure([gu.ErrorComponent("measurement", 0.3)])
        fresh = solver(
            only_translation=only_translation, subsample=60, tolerance_translation=1e-6, tolerance_rotation=1e-6
        )
        arguments: dict[str, Any] = dict(
            error_structure=changed_errors, method="analytical", random_state=12, at=["tx", "rz"]
        )
        expected = xdem.uncertainty.propagate_uncertainty_coreg(reference, changed_source, fresh, **arguments)
        summary = xdem.uncertainty.propagate_uncertainty_coreg(reference, changed_source, fitted, **arguments)

        # 3/ Compare the saved method's summary with a fresh fit using the supplied errors
        labels = ["tx", "ty", "tz", "rx", "ry", "rz"]
        expected_covariance = expected.output["parameter_covariance"]
        expected_std = expected.std
        covariance = summary.output["parameter_covariance"]

        # Check covariance labels, symmetry and standard deviations
        assert type(summary) is xdem.uncertainty.PropagationSummary
        assert list(covariance.index) == list(covariance.columns) == labels
        np.testing.assert_array_equal(covariance, expected_covariance)
        np.testing.assert_array_equal(summary.std, expected_std)
        np.testing.assert_allclose(np.diag(covariance), summary.std**2, rtol=1e-12, atol=1e-20)
        np.testing.assert_allclose(covariance, covariance.T, rtol=0, atol=1e-15)
        assert np.linalg.eigvalsh(covariance).min() >= -1e-15

        # Check the original fit, selected outputs and parameter estimates
        np.testing.assert_array_equal(summary.mean, summary.estimate)
        np.testing.assert_allclose(summary.estimate, expected.estimate)
        np.testing.assert_array_equal(fitted.to_matrix(), matrix)
        assert summary.output["centroid"] == fitted.centroid()
        assert list(summary.selection.index) == ["tx", "rz"]
        assert summary.coregs is None
        if solver is xdem.coreg.ICP:
            assert summary.output["fixed_correspondences"] is True

    @pytest.mark.parametrize("solver", [xdem.coreg.LZD, xdem.coreg.ICP])
    def test_propagate_uncertainty_coreg__refit_without_saved_errors(
        self, icp_dem: gu.Raster, solver: type[xdem.coreg.AffineCoreg]
    ) -> None:
        """Checks that a fit without saved uncertainty is refitted with the supplied error model."""

        # Fit without an error model and mark the old convergence result as unsuitable for reuse
        fitted = solver(only_translation=True, subsample=60).fit(icp_dem, icp_dem + 0.2, random_state=12)
        fitted.meta["outputs"]["iterative"]["converged"] = False
        original_matrix = fitted.to_matrix().copy()
        errors = gu.ErrorStructure([gu.ErrorComponent("measurement", 0.1)])

        # Calculate analytical uncertainty from a new fit to the supplied inputs
        summary = xdem.uncertainty.propagate_uncertainty_coreg(
            icp_dem, icp_dem + 0.2, fitted, error_structure=errors, method="analytical", random_state=12
        )

        # Check nonzero translation errors and preserve the original method's metadata
        assert np.all(summary.std.iloc[:3] > 0)
        np.testing.assert_array_equal(fitted.to_matrix(), original_matrix)
        assert not fitted.meta["outputs"]["iterative"]["converged"]
        assert "fit_pcov" not in fitted.meta["outputs"]["fitorbin"]

    @pytest.mark.parametrize("input_kind", ["raster", "pointcloud"])
    @pytest.mark.parametrize("error_applied_to", ["ref", "tba"])
    def test_propagate_uncertainty_coreg__xarray(
        self, input_kind: str, error_applied_to: Literal["ref", "tba"]
    ) -> None:
        """Checks that Dataset variables and native inputs give the same LZD uncertainty."""

        # 1/ Prepare inputs
        # Create curved terrain to constrain all translations
        rows, columns = np.mgrid[:15, :17]
        values = 5 * np.sin(columns / 7) + 3 * np.cos(rows / 6) + 2 * np.sin((columns + rows) / 11)
        transform = Affine(1, 0, 500_000, 0, -1, 7_000_000)
        reference = xdem.DEM(gu.Raster.from_array(values, transform, 32632))
        source = reference + 1
        if input_kind == "pointcloud":
            source = source.to_pointcloud()
        errors = xdem.ErrorStructure([xdem.ErrorComponent("measurement", 0.01)])
        coreg_method = xdem.coreg.LZD(
            only_translation=True, subsample=1, tolerance_translation=1e-6, tolerance_rotation=None
        )

        # Exclude edges so simulated horizontal shifts do not change the valid observation sample
        mask = np.zeros(reference.shape, dtype=bool)
        mask[1:-1, 1:-1] = True

        # Create a Dataset with separate raster and point coordinates
        source_array = source.to_xarray()
        if input_kind == "pointcloud":
            source_array = source_array.rename({"x": "x_point", "y": "y_point"})
        dataset = xr.Dataset({"reference": reference.to_xarray(), "elevation": source_array})
        original = dataset.copy(deep=True)
        options: dict[str, Any] = dict(
            error_structure={error_applied_to: errors},
            method="numerical",
            random_state=2,
            n_samples=3,
            return_samples=True,
            inlier_mask=mask,
        )

        # 2/ Run and compare outputs
        # Run propagation on native inputs and Dataset variables
        expected = xdem.uncertainty.propagate_uncertainty_coreg(reference, source, coreg_method, **options)
        actual = xdem.uncertainty.propagate_uncertainty_coreg(
            dataset["reference"], dataset["elevation"], coreg_method, **options
        )

        # Check equal summaries and samples
        pd.testing.assert_series_equal(actual.mean, expected.mean)
        pd.testing.assert_series_equal(actual.std, expected.std)
        assert actual.samples is not None and expected.samples is not None
        pd.testing.assert_frame_equal(actual.samples, expected.samples)

        # Check unchanged inputs
        xr.testing.assert_identical(dataset, original)
        assert not coreg_method._fit_called

    @pytest.mark.parametrize("crs, unit", [(32632, "metre"), (2227, "US survey foot")])
    def test_propagate_uncertainty_coreg__file_crs_units(self, tmp_path: Path, crs: int, unit: str) -> None:
        """Checks that a file-backed raster gives a parameter summary with its projected coordinate units."""

        # Write a projected raster and reopen it with the CRS supplied by Rasterio
        values = np.arange(48.0).reshape(6, 8)
        raster = gu.Raster.from_array(values, Affine(2, 0, 0, 0, -2, 12), crs)
        filename = tmp_path / "reference.tif"
        raster.to_file(filename)
        reference = gu.Raster(filename)
        errors = gu.ErrorStructure([gu.ErrorComponent("measurement", 0.2)])

        # Simulate the correction of a known vertical offset using the file-backed reference
        summary = xdem.uncertainty.propagate_uncertainty_coreg(
            reference,
            reference + 5,
            xdem.coreg.VerticalShift(vshift_reduc_func=np.mean),
            error_structure=errors,
            n_samples=4,
            random_state=2,
        )

        # Check that unit lookup succeeds without changing the CRS or the fitted correction
        assert summary.output["translation_units"] == unit
        assert summary.output["crs"] == reference.crs
        assert summary.estimate["tz"] == pytest.approx(-5)


class TestCoregistrationSimulation:
    """Test module for analytical uncertainty compared with numerical coregistration simulations."""

    @pytest.mark.parametrize("gls_method", ["approximate", "exact", "inducing"])
    @pytest.mark.parametrize("correlated", [False, True])
    @pytest.mark.parametrize("only_translation", [True, False])
    def test_propagate_uncertainty_coreg__gls_matches_simulation(
        self,
        monkeypatch: pytest.MonkeyPatch,
        correlated: bool,
        only_translation: bool,
        gls_method: str,
    ) -> None:
        """Checks that analytical GLS standard errors agree with repeated fits under the same elevation errors."""

        pytest.importorskip("torch")
        pytest.importorskip("gpytorch")

        from functools import partial

        from xdem.coreg.affine import (
            _gls_approximate,
            _gls_inducing,
            _gls_lazy_gpytorch,
        )

        # Force blockwise solves even for this small DEM
        if gls_method == "exact":
            monkeypatch.setattr(
                "xdem.coreg.affine._gls_lazy_gpytorch",
                partial(_gls_lazy_gpytorch, block_size=32, max_preconditioner_size=16),
            )
        elif gls_method == "approximate":
            monkeypatch.setattr("xdem.coreg.affine._gls_approximate", partial(_gls_approximate, block_size=32))
        else:
            monkeypatch.setattr("xdem.coreg.affine._gls_inducing", partial(_gls_inducing, block_size=32))

        # 1/ Create a small curved DEM and its error model
        # Curvature constrains all six parameters, and the border excludes observations that could leave the grid
        rows, columns = np.indices((12, 13))
        values = 5 * np.sin(columns / 3) + 3 * np.cos(rows / 4) + 2 * np.sin((columns + rows) / 5)
        reference = gu.Raster.from_array(values, Affine(2, 0, 0, 0, -2, 24), 32632)
        source = reference.copy()
        mask = np.zeros(reference.shape, dtype=bool)
        mask[2:-2, 2:-2] = True

        # Errors below the fit tolerances check the linear approximation before further resampling
        magnitude = 1e-5
        model = VariogramModel("gaussian", effective_range=4, partial_sill=1) if correlated else None
        component = gu.ErrorComponent("measurement", magnitude, correlation=model)
        errors = gu.ErrorStructure([component])
        correlation = (component.predict_correlation, 4) if correlated else None
        fit_options = {
            "sig_tba": np.full(reference.shape, magnitude),
            "corr_tba": correlation,
            "force_opti": "gls",
            "gls_method": gls_method,
            "gls_neighbors": 10,
            "gls_rank": 8,
        }
        options: dict[str, Any] = {
            "error_structure": errors,
            "inlier_mask": mask,
            "fit_kwargs": fit_options,
            "random_state": 42,
        }
        method = xdem.coreg.LZD(only_translation=only_translation, subsample=1)

        # 2/ Compare fitted standard errors with exact native Gaussian simulations
        _compare_simulated_uncertainty(
            reference, source, method, options, only_translation, n_samples=20, check_mean=True
        )

    @pytest.mark.parametrize("solver", [xdem.coreg.LZD, xdem.coreg.ICP])
    @pytest.mark.parametrize("only_translation", [True, False])
    @pytest.mark.parametrize("point_input", [None, "tba"])
    def test_propagate_uncertainty_coreg__both_inputs_match_simulation(
        self,
        icp_dem: gu.Raster,
        solver: Any,
        only_translation: bool,
        point_input: str | None,
    ) -> None:
        """Checks that both input models predict simulated translation and rotation errors."""

        pytest.importorskip("torch")
        pytest.importorskip("gpytorch")

        # 1/ Create two independent spatial error models with different magnitude predictors
        reference = icp_dem
        moving = reference + 0.2
        if point_input == "tba":
            # Use the raster's native coordinates so pixel interpretation does not add a horizontal shift
            x, y = reference.coords(grid=True)
            moving = gu.PointCloud.from_xyz(x.ravel(), y.ravel(), moving.data.ravel(), crs=reference.crs)
        sources = {"ref": reference, "tba": moving}
        models = {}
        predictors = {}
        for name, source in sources.items():
            quality = np.linspace(0, 1, source.data.size)
            predictor = "reference_quality" if name == "ref" else "moving_quality"
            magnitudes = [1e-4, 2e-4] if name == "ref" else [2e-4, 3e-4]
            statistics = pd.DataFrame(
                {"std": magnitudes, "count": [100, 100]}, index=pd.Index([0.0, 1.0], name=predictor)
            )
            magnitude = gu.ErrorMagnitude.variable_from_grouped_stats(statistics)
            correlation = VariogramModel("gaussian", effective_range=6 if name == "ref" else 4, partial_sill=1)
            model = gu.ErrorStructure(
                [
                    gu.ErrorComponent("measurement", 5e-5),
                    gu.ErrorComponent("terrain", magnitude, correlation),
                ]
            )
            models[name] = model
            predictors[name] = {predictor: quality.reshape(source.data.shape)}

        # 2/ Propagate both fields through identical analytical and numerical fits
        mask = np.zeros(icp_dem.shape, dtype=bool)
        mask[2:-2, 2:-2] = True
        method = solver(
            only_translation=only_translation,
            subsample=1,
            max_iterations=50,
            tolerance_translation=1e-7,
            tolerance_rotation=1e-7,
        )
        options: dict[str, Any] = dict(
            error_structure=models, error_predictors=predictors, inlier_mask=mask, random_state=42
        )
        analytical = _compare_simulated_uncertainty(reference, moving, method, options, only_translation)

        # Check the original models and all six covariance dimensions
        assert analytical.error_structure is models
        assert analytical.output["parameter_covariance"].shape == (6, 6)

    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    @pytest.mark.parametrize("layout", ["raster", "points", "inverted"])
    @pytest.mark.parametrize("only_translation", [True, False])
    @pytest.mark.parametrize("error_applied_to", ["ref", "tba"])
    @pytest.mark.parametrize("correlated", [False, True])
    def test_propagate_uncertainty_coreg__icp_matches_simulation(
        self,
        icp_dem: gu.Raster,
        layout: str,
        only_translation: bool,
        error_applied_to: Literal["ref", "tba"],
        correlated: bool,
    ) -> None:
        """Checks that analytical ICP translation and rotation errors agree with exact native Gaussian simulations."""

        pytest.importorskip("torch")
        pytest.importorskip("gpytorch")

        # 1/ Create stable correspondences with a nonzero transform and an interior inlier mask
        # A raster shift requires interpolation, while transformed native points preserve exact geometry
        reference = icp_dem
        if layout == "raster":
            moving = reference.translate(0.15, -0.12) + 0.1
        else:
            x, y = reference.coords(grid=True)
            points = gu.PointCloud.from_xyz(x.ravel(), y.ravel(), reference.data.ravel(), crs=reference.crs)
            angles = (0, 0, 0) if only_translation else (0.12, -0.09, 0.15)
            planted = xdem.coreg.matrix_from_translations_rotations(0.18, -0.12, 0.24, *angles)
            center = (float(x.mean()), float(y.mean()), float(reference.data.mean()))
            transformed = xdem.coreg.apply_matrix(points.gdf, planted, centroid=center)
            moving = gu.PointCloud(transformed, data_name="z")
            if layout == "inverted":
                reference, moving = moving, reference
        mask = np.zeros(icp_dem.shape, dtype=bool)
        mask[2:-2, 2:-2] = True
        method = xdem.coreg.ICP(
            only_translation=only_translation,
            subsample=1,
            sampling_strategy="same_xy" if layout == "raster" else "independent",
            tolerance_translation=1e-9,
            tolerance_rotation=1e-8,
            max_iterations=50,
        )
        magnitude = 1e-4
        model = VariogramModel("gaussian", effective_range=6, partial_sill=1) if correlated else None
        component = gu.ErrorComponent("measurement", magnitude, correlation=model)
        errors = gu.ErrorStructure([component])

        # 2/ Draw native Gaussian fields independently of the analytical covariance implementation
        options: dict[str, Any] = dict(error_structure={error_applied_to: errors}, inlier_mask=mask, random_state=42)
        _compare_simulated_uncertainty(reference, moving, method, options, only_translation, check_mean=True)


class TestCoregistrationPropagationErrors:
    """Test module for unsupported coregistration uncertainty inputs, stored fits and failed simulations."""

    def test_propagate_uncertainty_coreg__error_analytical_coregs(self) -> None:
        """Checks an error is raised for simulated Coregs requested with analytical propagation."""

        # Create an error model
        errors = gu.ErrorStructure([gu.ErrorComponent("measurement", 0.1)])
        method = xdem.coreg.VerticalShift()

        # Check analytical propagation cannot return simulated Coregs
        with pytest.raises(ValueError, match="Analytical propagation cannot return simulated Coreg objects"):
            xdem.uncertainty.propagate_uncertainty_coreg(
                None, None, method, error_structure=errors, method="analytical", return_coregs=True
            )

    @pytest.mark.parametrize("stage", ["draw", "fit"])
    def test_propagate_uncertainty_coreg__error_failed_simulation(
        self, monkeypatch: pytest.MonkeyPatch, stage: str
    ) -> None:
        """Checks that failed draws or fits are recorded and excluded from summaries and saved Coregs."""

        # 1/ Prepare a failure in the first simulation, after the original fit when applicable
        reference = gu.Raster.from_array(np.arange(48, dtype=float).reshape(6, 8), Affine(2, 0, 0, 0, -2, 12), 32632)
        errors = gu.ErrorStructure([gu.ErrorComponent("measurement", 0.2)])
        owner = gu.ErrorStructure if stage == "draw" else xdem.coreg.VerticalShift
        name = "generate_random_field" if stage == "draw" else "fit"
        original = getattr(owner, name)
        calls = iter([True, False, True, True] if stage == "fit" else [False, True, True])
        message = f"failed {stage}"

        def fail_first_simulation(self: Any, *args: Any, **kwargs: Any) -> Any:
            """Fail the first simulation and otherwise use the original draw or fit."""

            if not next(calls):
                raise RuntimeError(message)
            return original(self, *args, **kwargs)

        # 2/ Run simulations and check the failure warning
        monkeypatch.setattr(owner, name, fail_first_simulation)
        with pytest.warns(UserWarning, match=message):
            summary = xdem.uncertainty.propagate_uncertainty_coreg(
                reference,
                reference + 5,
                xdem.coreg.VerticalShift(vshift_reduc_func=np.mean),
                error_structure=errors,
                n_samples=3,
                random_state=2,
                return_coregs=True,
                return_samples=True,
            )

        # 3/ Compare saved Coregs, sample rows and uncertainty from successful fits
        assert summary.n_samples == 3
        assert summary.n_success == 2
        assert summary.failures == {1: message}
        assert summary.samples is not None
        assert summary.samples.index.tolist() == [1, 2, 3]
        assert summary.samples.loc[1].isna().all()
        assert summary.coregs is not None
        assert list(summary.coregs) == [2, 3]
        for number, fitted in summary.coregs.items():
            assert fitted.to_matrix()[2, 3] == summary.samples.loc[number, "tz"]
        np.testing.assert_allclose(summary.std, summary.samples.std())

    @pytest.mark.parametrize(
        "options",
        [
            {"method": "point-to-point"},
            {"fit_loss_func": "soft_l1"},
            {"trim_residuals": True},
            {"linearized": False},
            {"sampling_strategy": "iterative_same_xy"},
        ],
    )
    def test_propagate_uncertainty_coreg__error_icp_options(self, options: dict[str, Any]) -> None:
        """Checks an error is raised for unsupported ICP uncertainty options before reading elevations."""

        # Create a supported error model and an ICP option that changes the analytical estimator
        method = xdem.coreg.ICP(**options)
        errors = gu.ErrorStructure([gu.ErrorComponent("measurement", 0.1)])

        # Request analytical uncertainty before supplying any elevations
        with pytest.raises(NotImplementedError, match="Analytical ICP"):
            xdem.uncertainty.propagate_uncertainty_coreg(
                None, None, method, error_structure=errors, method="analytical"
            )

    @pytest.mark.parametrize(
        "options, message",
        [
            ({"error_structure": {}}, "nonempty mapping"),
            ({"error_structure": {"moving": None}}, "keys must"),
            ({"error_structure": {"tba": None}}, "Each error_structure value"),
            ({"error_predictors": {"ref": {}}}, "keys must name inputs"),
            ({"error_predictors": {"tba": []}}, "Each error_predictors value"),
        ],
    )
    def test_propagate_uncertainty_coreg__error_invalid_mapping(self, options: dict[str, Any], message: str) -> None:
        """Checks an error is raised for invalid mappings before reading elevation inputs."""

        # Create default moving errors and replace one mapping with invalid values
        errors = gu.ErrorStructure([gu.ErrorComponent("measurement", 0.01)])
        options = {"error_structure": {"tba": errors}, **options}

        # Check validation before fitting or resolving either elevation
        with pytest.raises((TypeError, ValueError), match=message):
            xdem.uncertainty.propagate_uncertainty_coreg(None, None, xdem.coreg.LZD(), **options)
