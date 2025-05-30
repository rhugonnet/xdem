"""Functions to test the coregistration workflows."""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pandas as pd
import pytest
from geoutils import Raster, Vector
from geoutils.raster import RasterType
from geoutils.stats import nmad

import xdem
from xdem import examples
from xdem.coreg.workflows import create_inlier_mask, dem_coregistration


def load_examples() -> tuple[RasterType, RasterType, Vector]:
    """Load example files to try coregistration methods with."""

    reference_raster = Raster(examples.get_path("longyearbyen_ref_dem"))
    to_be_aligned_raster = Raster(examples.get_path("longyearbyen_tba_dem"))
    glacier_mask = Vector(examples.get_path("longyearbyen_glacier_outlines"))

    return reference_raster, to_be_aligned_raster, glacier_mask


class TestWorkflows:
    def test_create_inlier_mask(self) -> None:
        """Test that the create_inlier_mask function works expectedly."""

        ref, tba, outlines = load_examples()  # Load example reference, to-be-aligned and outlines

        # - Assert that without filtering create_inlier_mask behaves as if calling Vector.create_mask - #
        # Masking inside - using Vector
        inlier_mask_comp = ~outlines.create_mask(ref, as_array=True)
        inlier_mask = create_inlier_mask(
            tba,
            ref,
            [
                outlines,
            ],
            filtering=False,
        )
        assert np.all(inlier_mask_comp == inlier_mask)

        # Masking inside - using string
        inlier_mask = create_inlier_mask(
            tba,
            ref,
            [
                outlines.name,
            ],
            filtering=False,
        )
        assert np.all(inlier_mask_comp == inlier_mask)

        # Masking outside - using Vector
        inlier_mask = create_inlier_mask(
            tba,
            ref,
            [
                outlines,
            ],
            inout=[
                -1,
            ],
            filtering=False,
        )
        assert np.all(~inlier_mask_comp == inlier_mask)

        # Masking outside - using string
        inlier_mask = create_inlier_mask(
            tba,
            ref,
            [
                outlines.name,
            ],
            inout=[-1],
            filtering=False,
        )
        assert np.all(~inlier_mask_comp == inlier_mask)

        # - Test filtering options only - #
        # Test the slope filter only
        slope = xdem.terrain.slope(ref)
        slope_lim = [1, 50]
        inlier_mask_comp2 = np.ones(tba.data.shape, dtype=bool)
        inlier_mask_comp2[slope.data < slope_lim[0]] = False
        inlier_mask_comp2[slope.data > slope_lim[1]] = False
        inlier_mask = create_inlier_mask(tba, ref, filtering=True, slope_lim=slope_lim, nmad_factor=np.inf)
        assert np.all(inlier_mask == inlier_mask_comp2)

        # Test the nmad_factor filter only
        nmad_factor = 3
        ddem = tba - ref
        inlier_mask_comp3 = (np.abs(ddem.data - np.median(ddem)) < nmad_factor * nmad(ddem.data)).filled(False)
        inlier_mask = create_inlier_mask(tba, ref, filtering=True, slope_lim=[0, 90], nmad_factor=nmad_factor)
        assert np.all(inlier_mask == inlier_mask_comp3)

        # Test the sum of both
        inlier_mask = create_inlier_mask(
            tba, ref, shp_list=[], inout=[], filtering=True, slope_lim=slope_lim, nmad_factor=nmad_factor
        )
        inlier_mask_all = inlier_mask_comp2 & inlier_mask_comp3
        assert np.all(inlier_mask == inlier_mask_all)

        # Test the dh_max filter only
        dh_max = 200
        inlier_mask_comp4 = (np.abs(ddem.data) < dh_max).filled(False)
        inlier_mask = create_inlier_mask(tba, ref, filtering=True, slope_lim=[0, 90], nmad_factor=np.inf, dh_max=dh_max)
        assert np.all(inlier_mask == inlier_mask_comp4)

        # - Test the sum of outlines + dh_max + slope - #
        # nmad_factor will have a different behavior because it calculates nmad from the inliers of previous filters
        inlier_mask = create_inlier_mask(
            tba,
            ref,
            shp_list=[
                outlines,
            ],
            inout=[
                -1,
            ],
            filtering=True,
            slope_lim=slope_lim,
            nmad_factor=np.inf,
            dh_max=dh_max,
        )
        inlier_mask_all = ~inlier_mask_comp & inlier_mask_comp2 & inlier_mask_comp4
        assert np.all(inlier_mask == inlier_mask_all)

        # - Test that proper errors are raised for wrong inputs - #
        with pytest.raises(ValueError, match="`shp_list` must be a list/tuple"):
            create_inlier_mask(tba, ref, shp_list=outlines)

        with pytest.raises(ValueError, match="`shp_list` must be a list/tuple of strings or geoutils.Vector instance"):
            create_inlier_mask(tba, ref, shp_list=[1])

        with pytest.raises(ValueError, match="`inout` must be a list/tuple"):
            create_inlier_mask(
                tba,
                ref,
                shp_list=[
                    outlines,
                ],
                inout=1,  # type: ignore
            )

        with pytest.raises(ValueError, match="`inout` must contain only 1 and -1"):
            create_inlier_mask(
                tba,
                ref,
                shp_list=[
                    outlines,
                ],
                inout=[
                    0,
                ],
            )

        with pytest.raises(ValueError, match="`inout` must be of same length as shp"):
            create_inlier_mask(
                tba,
                ref,
                shp_list=[
                    outlines,
                ],
                inout=[1, 1],
            )

        with pytest.raises(ValueError, match="`slope_lim` must be a list/tuple"):
            create_inlier_mask(tba, ref, filtering=True, slope_lim=1)  # type: ignore

        with pytest.raises(ValueError, match="`slope_lim` must contain 2 elements"):
            create_inlier_mask(tba, ref, filtering=True, slope_lim=[30])

        with pytest.raises(ValueError, match=r"`slope_lim` must be a tuple/list of 2 elements in the range \[0-90\]"):
            create_inlier_mask(tba, ref, filtering=True, slope_lim=[-1, 40])

        with pytest.raises(ValueError, match=r"`slope_lim` must be a tuple/list of 2 elements in the range \[0-90\]"):
            create_inlier_mask(tba, ref, filtering=True, slope_lim=[1, 120])

    def test_dem_coregistration(self) -> None:
        """
        Test that the dem_coregistration function works expectedly.
        Tests the features that are specific to dem_coregistration.
        For example, many features are tested in create_inlier_mask, so not tested again here.
        TODO: Add DEMs with different projection/grid to test that regridding works as expected.
        """
        # Load example reference, to-be-aligned and outlines
        ref_dem, tba_dem, outlines = load_examples()

        # - Check that it works with default parameters - #
        dem_coreg, coreg_method, coreg_stats, inlier_mask = dem_coregistration(tba_dem, ref_dem, random_state=42)

        # Assert that outputs have expected format
        assert isinstance(dem_coreg, xdem.DEM)
        assert isinstance(coreg_method, xdem.coreg.Coreg)
        assert isinstance(coreg_stats, pd.DataFrame)

        # Assert that default coreg_method is as expected
        assert hasattr(coreg_method, "pipeline")
        assert isinstance(coreg_method.pipeline[0], xdem.coreg.NuthKaab)
        assert isinstance(coreg_method.pipeline[1], xdem.coreg.VerticalShift)

        # The result should be similar to applying the same coreg by hand with:
        # - DEMs converted to Float32
        # - default inlier_mask
        # - no resampling
        coreg_method_ref = xdem.coreg.NuthKaab() + xdem.coreg.VerticalShift()
        inlier_mask = create_inlier_mask(tba_dem, ref_dem)
        coreg_method_ref.fit(
            ref_dem.astype(np.float32), tba_dem.astype(np.float32), inlier_mask=inlier_mask, random_state=42
        )
        dem_coreg_ref = coreg_method_ref.apply(tba_dem.astype(np.float32), resample=False)
        assert dem_coreg.raster_equal(dem_coreg_ref, warn_failure_reason=True)

        # Assert that coregistration improved the residuals
        assert abs(coreg_stats["med_orig"].values) > abs(coreg_stats["med_coreg"].values)
        assert coreg_stats["nmad_orig"].values > coreg_stats["nmad_coreg"].values

        # - Check some alternative arguments - #
        # Test with filename instead of DEMs
        dem_coreg2, _, _, _ = dem_coregistration(tba_dem.filename, ref_dem.filename, random_state=42)
        assert dem_coreg2.raster_equal(dem_coreg, warn_failure_reason=True)

        # Test saving to file (mode = "w" is necessary to work on Windows)
        outfile = tempfile.NamedTemporaryFile(suffix=".tif", mode="w", delete=False)
        dem_coregistration(tba_dem, ref_dem, out_dem_path=outfile.name, random_state=42)
        dem_coreg2 = xdem.DEM(outfile.name)
        assert dem_coreg2.raster_equal(dem_coreg, warn_failure_reason=True)
        outfile.close()

        # Test that shapefile is properly taken into account - inlier_mask should be False inside outlines
        # Need to use resample=True, to ensure that dem_coreg has same georef as inlier_mask
        dem_coreg, coreg_method, coreg_stats, inlier_mask = dem_coregistration(
            tba_dem,
            ref_dem,
            shp_list=[
                outlines,
            ],
            resample=True,
        )
        gl_mask = outlines.create_mask(dem_coreg, as_array=True)
        assert np.all(~inlier_mask[gl_mask])

        # Testing with plot
        out_fig = tempfile.NamedTemporaryFile(suffix=".png", mode="w", delete=False)
        assert os.path.getsize(out_fig.name) == 0
        dem_coregistration(tba_dem, ref_dem, plot=True, out_fig=out_fig.name)
        assert os.path.getsize(out_fig.name) > 0
        out_fig.close()

        # Testing different coreg method
        dem_coreg2, coreg_method2, coreg_stats2, inlier_mask2 = dem_coregistration(
            tba_dem, ref_dem, coreg_method=xdem.coreg.Deramp()
        )
        assert isinstance(coreg_method2, xdem.coreg.Deramp)
        assert abs(coreg_stats2["med_orig"].values) > abs(coreg_stats2["med_coreg"].values)
        assert coreg_stats2["nmad_orig"].values > coreg_stats2["nmad_coreg"].values

        # Testing with initial shift
        test_shift_list = [10, 5]
        tba_dem_origin = tba_dem.copy()
        coreg_pipeline = xdem.coreg.affine.NuthKaab() + xdem.coreg.affine.VerticalShift()

        dem_coreg2, coreg_method2, coreg_stats2, inlier_mask2 = dem_coregistration(
            tba_dem, ref_dem, coreg_method=coreg_pipeline, estimated_initial_shift=test_shift_list, random_state=42
        )
        dem_coreg3, coreg_method3, coreg_stats3, inlier_mask3 = dem_coregistration(
            tba_dem, ref_dem, coreg_method=coreg_pipeline, random_state=42
        )
        assert tba_dem.raster_equal(tba_dem_origin)
        assert isinstance(coreg_method2, xdem.coreg.CoregPipeline)
        assert isinstance(coreg_method3, xdem.coreg.CoregPipeline)
        assert isinstance(coreg_method2.pipeline[0], xdem.coreg.AffineCoreg)
        assert isinstance(coreg_method3.pipeline[0], xdem.coreg.AffineCoreg)
        assert (
            coreg_method2.pipeline[0].meta["outputs"]["affine"]["shift_x"]
            == coreg_method3.pipeline[0].meta["outputs"]["affine"]["shift_x"]
        )
        assert (
            coreg_method2.pipeline[0].meta["outputs"]["affine"]["shift_y"]
            == coreg_method3.pipeline[0].meta["outputs"]["affine"]["shift_y"]
        )

        # Testing without coreg pipeline
        test_shift_tuple = (-5, 2)  # tuple
        coreg_simple = xdem.coreg.affine.DhMinimize()

        dem_coreg2, coreg_method2, coreg_stats2, inlier_mask2 = dem_coregistration(
            tba_dem, ref_dem, coreg_method=coreg_simple, estimated_initial_shift=test_shift_tuple, random_state=42
        )
        dem_coreg3, coreg_method3, coreg_stats3, inlier_mask3 = dem_coregistration(
            tba_dem, ref_dem, coreg_method=coreg_simple, random_state=42
        )
        assert isinstance(coreg_method2, xdem.coreg.AffineCoreg)
        assert isinstance(coreg_method3, xdem.coreg.AffineCoreg)
        assert coreg_method2.meta["outputs"]["affine"]["shift_x"] == coreg_method3.meta["outputs"]["affine"]["shift_x"]
        assert coreg_method2.meta["outputs"]["affine"]["shift_y"] == coreg_method3.meta["outputs"]["affine"]["shift_y"]

        # Check if the appropriate exception is raised with an initial shift and without affine coreg
        with pytest.raises(TypeError, match=r".*affine.*"):
            dem_coregistration(
                tba_dem,
                ref_dem,
                coreg_method=xdem.coreg.Deramp(),
                estimated_initial_shift=test_shift_tuple,
                random_state=42,
            )
        with pytest.raises(TypeError, match=r".*affine.*"):
            dem_coregistration(
                tba_dem,
                ref_dem,
                coreg_method=xdem.coreg.Deramp() + xdem.coreg.TerrainBias(),
                estimated_initial_shift=test_shift_tuple,
                random_state=42,
            )

        # Check if the appropriate exception is raised with a wrong type initial shift
        with pytest.raises(ValueError, match=r".*two numerical values.*"):
            dem_coregistration(
                tba_dem,
                ref_dem,
                estimated_initial_shift=["2", 2],
                random_state=42,
            )
        with pytest.raises(ValueError, match=r".*two numerical values.*"):
            dem_coregistration(
                tba_dem,
                ref_dem,
                estimated_initial_shift=[2, 3, 5],
                random_state=42,
            )
