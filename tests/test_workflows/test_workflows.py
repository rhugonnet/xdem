# Copyright (c) 2025 Centre National d'Etudes Spatiales (CNES).
#
# This file is part of the xDEM project:
# https://github.com/glaciohack/xdem
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""

test for workflow class
"""

import re
from pathlib import Path
from typing import Any, Dict

import geoutils as gu
import pytest
import rasterio
import xarray as xr

import xdem
from xdem.workflows.accuracy import Accuracy
from xdem.workflows.schemas import (
    COMPLETE_CONFIG_ACCURACY,
    COMPLETE_CONFIG_TOPO,
    MIN_STATS,
    TERRAIN_ATTRIBUTES_DEFAULT,
)
from xdem.workflows.topo import Topo
from xdem.workflows.workflows import Workflows

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")

pytest.importorskip("cerberus")

import yaml  # type: ignore  # noqa


def test_workflows_init_wrong_config() -> None:
    """
    Test workflows class initialization
    """
    user_config = 2
    with pytest.raises(ValueError, match="The configuration should be provided either as a path"):
        _ = Topo(user_config)  # type: ignore


@pytest.mark.parametrize("level", [1, 2])
def test_workflows_init(get_topo_inputs_config_list: list[dict[str, str]], tmp_path: Path, level: int) -> None:
    """
    Test workflows class initialization
    """
    user_config: dict[str, Any] = dict()
    user_config["inputs"] = get_topo_inputs_config_list[:]
    user_config["outputs"] = {"path": str(tmp_path), "level": level}
    workflows = Topo(user_config)
    workflows.run()

    assert isinstance(workflows, Workflows)
    assert isinstance(workflows, Topo)


@pytest.mark.parametrize("subdir_given", [True, False])
def test_create_output_dir(
    get_topo_inputs_config_list: list[dict[str, str]], tmp_path: Path, subdir_given: bool
) -> None:
    """Create output directories at the configured or overridden path."""
    user_config: dict[str, Any] = dict()
    user_config["inputs"] = get_topo_inputs_config_list[:1]
    user_config["outputs"] = {"path": str(tmp_path)}
    workflow = Topo(user_config)
    target = tmp_path / "sub_dir" / "output" if subdir_given else tmp_path
    workflow.create_output_dir(target if subdir_given else None)

    assert workflow.outputs_folder == target
    assert all((target / name).is_dir() for name in ("plots", "rasters", "tables"))
    workflow.create_output_dir()
    assert workflow.outputs_folder == target
    assert all((target / name).is_dir() for name in ("plots", "rasters", "tables"))


def test_configure_execution(get_topo_inputs_config_list: list[dict[str, str]], tmp_path: Path) -> None:
    """Test configuration for the scalability mode."""
    user_config: dict[str, Any] = dict()
    user_config["inputs"] = get_topo_inputs_config_list[:1]
    user_config["outputs"] = {"path": str(tmp_path)}
    user_config["scalability"] = None
    workflow = Topo(user_config)
    workflow.configure_execution()
    workflow.configure_execution()
    assert workflow.dask is None
    assert workflow.mp_config is None

    user_config["scalability"] = {"dask": {"chunks": {"x": 20, "y": 30}}}
    workflow = Topo(user_config)
    workflow.configure_execution()
    assert workflow.dask is not None
    assert workflow.dask["chunks"] == {"x": 20, "y": 30, "band": 1}
    assert workflow.mp_config is None


@pytest.mark.parametrize(
    "input_data,expected",
    [
        ({"a": 1, "b": None, "c": 3}, {"a": 1, "c": 3}),
        ({"a": 1, "statistics": None, "c": 3}, {"a": 1, "statistics": None, "c": 3}),
        ({"a": {"x": None, "y": 2}, "b": None}, {"a": {"y": 2}}),
        ([1, None, 2, None, 3], [1, 2, 3]),
        ([{"a": 1, "b": None}, {"c": None}, None, {"d": 4}], [{"a": 1}, {}, {"d": 4}]),
        (
            {"a": [1, None, {"b": None, "c": 3}], "d": None, "e": {"f": None, "g": [None, 7]}},
            {"a": [1, {"c": 3}], "e": {"g": [7]}},
        ),
        ({}, {}),
        ([], []),
        ({"a": 1, "b": [2, 3], "c": {"d": 4}}, {"a": 1, "b": [2, 3], "c": {"d": 4}}),
    ],
)
def test_remove_none_cases(
    get_topo_inputs_config_list: list[dict[str, str]], input_data: Any, expected: Any, tmp_path: Path
) -> None:
    """
    Test remove_none from dictionary
    """
    user_config: dict[str, Any] = dict()
    user_config["inputs"] = get_topo_inputs_config_list[:1]
    user_config["outputs"] = {"path": str(tmp_path)}
    workflows = Topo(user_config)
    assert workflows.remove_none(input_data) == expected


def test_load_config(get_topo_inputs_config_list: list[dict[str, str]], tmp_path: Path) -> None:
    """
    Test load_config function
    """
    user_config: dict[str, Any] = dict()
    user_config["inputs"] = get_topo_inputs_config_list[:1]
    user_config["outputs"] = {"path": str(tmp_path)}

    # Succeed
    with open(tmp_path / "temp_config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(user_config, f, allow_unicode=True, default_flow_style=False)

    workflows = Topo(str(tmp_path / "temp_config.yaml"))
    assert workflows.load_config() == user_config

    # Fail
    missing_config = tmp_path / "tempconfig.yaml"
    with pytest.raises(FileNotFoundError, match=re.escape(f"{missing_config} does not exist")):
        _ = Topo(str(missing_config))


def test_load_config_none(
    get_topo_inputs_config_list: list[dict[str, str]], get_accuracy_inputs_test: Dict[str, Any], tmp_path: Path
) -> None:
    """
    Test None values in yaml reading function
    """
    # Change values
    user_config: dict[str, Any] = dict()
    user_config["inputs"] = get_topo_inputs_config_list[:1]
    user_config["inputs"][0]["force_vcrs"] = None
    user_config["outputs"] = {"path": str(tmp_path)}

    # Read working config
    yaml_str = yaml.dump(user_config, allow_unicode=True)
    Path(tmp_path / "temp_config.yaml").write_text(yaml_str, encoding="utf-8")
    workflow_topo = Topo(str(tmp_path / "temp_config.yaml"))
    assert isinstance(workflow_topo, Workflows)
    assert isinstance(workflow_topo, Topo)
    config_output = workflow_topo.load_config()
    assert config_output["inputs"][0]["force_vcrs"] is None

    # Accuracy workflow

    # Change values
    cfg = get_accuracy_inputs_test
    cfg["inputs"]["reference_elev"]["force_vcrs"] = "None"
    cfg["inputs"]["sampling_grid"] = "None"
    cfg["coregistration"] = {}
    cfg["coregistration"]["process"] = False

    # Read working config
    yaml_str = yaml.dump(cfg, allow_unicode=True)
    Path(tmp_path / "temp_config.yaml").write_text(yaml_str, encoding="utf-8")
    workflow_accuracy = Accuracy(str(tmp_path / "temp_config.yaml"))
    assert isinstance(workflow_accuracy, Workflows)
    assert isinstance(workflow_accuracy, Accuracy)
    config_output = workflow_accuracy.load_config()
    assert config_output["inputs"]["reference_elev"]["force_vcrs"] is None
    assert config_output["inputs"]["sampling_grid"] is None


def test_pipeline_accuracy_default_values(get_accuracy_inputs_test: Dict[str, Any], tmp_path: Path) -> None:
    """
    Test valid VCRS function for 'from' and 'to'
    """
    accuracy_config = get_accuracy_inputs_test
    yaml_str = yaml.dump(accuracy_config, allow_unicode=True)

    Path(tmp_path / "temp_config.yaml").write_text(yaml_str, encoding="utf-8")
    workflow_accuracy = Accuracy(str(tmp_path / "temp_config.yaml"))
    assert isinstance(workflow_accuracy, Workflows)
    assert isinstance(workflow_accuracy, Accuracy)
    pipeline_accuracy_test = workflow_accuracy.config

    for elev in ["reference_elev", "to_be_aligned_elev"]:
        input_elev = pipeline_accuracy_test["inputs"][elev]
        input_elev_input = accuracy_config["inputs"][elev]
        assert input_elev["path_to_elev"] == input_elev_input["path_to_elev"]
        if "path_to_mask" in input_elev_input:
            assert input_elev["path_to_mask"] == input_elev_input["path_to_mask"]

        assert "force_vcrs" not in input_elev
        assert input_elev["downsample"] == 1

    assert pipeline_accuracy_test["inputs"]["sampling_grid"] == "reference_elev"

    pipeline_corg = pipeline_accuracy_test["coregistration"]
    assert list(pipeline_corg.keys()) == ["step_one", "process"]
    assert pipeline_corg["step_one"] == COMPLETE_CONFIG_ACCURACY["coregistration"]["step_one"]  # type: ignore
    assert pipeline_accuracy_test["coregistration"]["process"]

    assert pipeline_accuracy_test["statistics"] == COMPLETE_CONFIG_ACCURACY["statistics"]
    assert pipeline_accuracy_test["outputs"] == COMPLETE_CONFIG_ACCURACY["outputs"]


def test_pipeline_topo_default_values(get_topo_inputs_config_list: list[dict[str, str]], tmp_path: Path) -> None:
    """
    Test valid VCRS function for 'from' and 'to'
    """
    user_config: dict[str, Any] = dict()
    user_config["inputs"] = get_topo_inputs_config_list[:1]
    user_config["inputs"][0]["force_vcrs"] = None
    user_config["outputs"] = {"path": str(tmp_path)}
    yaml_str = yaml.dump(user_config, allow_unicode=True)
    Path(tmp_path / "temp_config.yaml").write_text(yaml_str, encoding="utf-8")
    workflow_topo = Topo(str(tmp_path / "temp_config.yaml"))
    assert isinstance(workflow_topo, Workflows)
    assert isinstance(workflow_topo, Topo)
    pipeline_topo_test = workflow_topo.config

    assert len(pipeline_topo_test["inputs"]) == len(user_config["inputs"])
    for k, input_elev in enumerate(pipeline_topo_test["inputs"]):
        assert input_elev["path_to_elev"] == user_config["inputs"][k]["path_to_elev"]
        if "path_to_mask" in user_config["inputs"][k]:
            assert input_elev["path_to_mask"] == user_config["inputs"][k]["path_to_mask"]

        assert "force_vcrs" not in input_elev
        assert "downsample" not in input_elev  # default value not taken in "anyof" schema

    assert pipeline_topo_test["statistics"] == MIN_STATS
    assert pipeline_topo_test["terrain_attributes"] == TERRAIN_ATTRIBUTES_DEFAULT
    assert pipeline_topo_test["outputs"] == COMPLETE_CONFIG_TOPO["outputs"]


@pytest.mark.parametrize(
    "data, mask_type",
    [
        pytest.param("giza_dem", "none"),
        pytest.param("longyearbyen_ref_dem", "none"),
        pytest.param("longyearbyen_ref_dem", "vector"),
        pytest.param("longyearbyen_ref_dem", "raster"),
    ],
)
@pytest.mark.parametrize("force_vcrs", [None, "Ellipsoid", "EGM96"])
@pytest.mark.parametrize(
    "dask", [None, {"chunks": {"x": 200, "y": 200}}, {"chunks": {"x": 20, "y": 20}}, {"chunks": {"x": 13, "y": 17}}]
)
def test_load_dem(
    data: str, mask_type: str, force_vcrs: str | None, dask: Dict[str, Any] | None, tmp_path: Path
) -> None:
    config_dem: dict[str, Any] = {}
    config_dem["path_to_elev"] = (
        xdem.examples.get_path(data) if data == "giza_dem" else xdem.examples.get_path_test(data)
    )

    if mask_type == "vector":
        config_dem["path_to_mask"] = xdem.examples.get_path_test("longyearbyen_glacier_outlines")
    elif mask_type == "raster":
        vector_path = xdem.examples.get_path_test("longyearbyen_glacier_outlines")
        mask_file = tmp_path / "mask.tif"
        (~gu.Vector(vector_path).create_mask(xdem.DEM(config_dem["path_to_elev"]))).to_file(mask_file)
        config_dem["path_to_mask"] = str(mask_file)

    if force_vcrs is not None:
        config_dem["force_vcrs"] = force_vcrs

    config_dem["scalability"] = {"dask": dask} if dask is not None else None
    output_dem, inlier_mask, mask_path = Workflows.load_dem(config_dem, dask)

    if dask is None:
        assert isinstance(output_dem, xdem.DEM)
        assert isinstance(inlier_mask, gu.Raster) if "path_to_mask" in config_dem else inlier_mask is None
        assert isinstance(mask_path, str) if "path_to_mask" in config_dem else mask_path is None
        assert not output_dem.is_loaded

    else:
        import dask.array as da

        assert isinstance(output_dem, xr.core.dataarray.DataArray)
        assert isinstance(output_dem.data, da.Array)
        assert not output_dem.dem.is_loaded
        if "path_to_mask" in config_dem:
            assert isinstance(inlier_mask, xr.core.dataarray.DataArray)
            assert isinstance(inlier_mask.data, da.Array)
            assert not inlier_mask.rst.is_loaded
            assert isinstance(mask_path, str)
            assert inlier_mask.dtype == bool
        else:
            assert inlier_mask is None
            assert mask_path is None
        output_dem = output_dem.dem

    dem = xdem.DEM(config_dem["path_to_elev"])

    # VCRS
    if force_vcrs is None:
        assert output_dem.vcrs == dem.vcrs
    else:
        dem.set_vcrs(force_vcrs)
        assert output_dem.vcrs == dem.vcrs

    # DEM
    assert dem.georeferenced_grid_equal(output_dem)


@pytest.mark.parametrize("dem_alias", [True, False])
@pytest.mark.parametrize("mask_alias", [True, False])
@pytest.mark.parametrize("chunks", [None, {"chunks": {"x": 200, "y": 200}}])
def test_load_dem_alias(chunks: Dict[str, Any] | None, dem_alias: bool, mask_alias: bool) -> None:
    """
    Test load_dem function with alias
    """

    # Test with no mask
    config_dem = dict()

    dem_name = "longyearbyen_ref_dem"
    config_dem["path_to_elev"] = dem_name if dem_alias else xdem.examples.get_path(dem_name)
    output_dem, inlier_mask, mask_path = Workflows.load_dem(config_dem, chunks)

    trust_dem = xdem.DEM(xdem.examples.get_path(dem_name))
    if chunks is None:
        assert isinstance(output_dem, xdem.DEM)
        assert not output_dem.is_loaded
        assert output_dem.raster_equal(trust_dem)
    else:
        assert isinstance(output_dem, xr.core.dataarray.DataArray)
        assert not output_dem.dem.is_loaded
        assert output_dem.dem.raster_equal(trust_dem)

    assert inlier_mask is None
    assert mask_path is None

    config_dem["path_to_elev"] = "aliasnotexist"
    with pytest.raises(rasterio.errors.RasterioIOError, match="No such file or directory"):
        Workflows.load_dem(config_dem, chunks)

    # Test with mask

    config_dem["path_to_elev"] = dem_name if dem_alias else xdem.examples.get_path(dem_name)
    mask_name = "longyearbyen_glacier_outlines"
    config_dem["path_to_mask"] = mask_name if mask_alias else xdem.examples.get_path(mask_name)
    output_dem, inlier_mask, mask_path = Workflows.load_dem(config_dem, chunks)
    trust_mask = ~gu.Vector(xdem.examples.get_path(mask_name)).create_mask(output_dem)
    if chunks is None:
        assert isinstance(output_dem, xdem.DEM)
        assert not output_dem.is_loaded
        assert output_dem.raster_equal(trust_dem)
        assert isinstance(inlier_mask, gu.Raster)
        assert inlier_mask.raster_equal(trust_mask)
    else:
        assert isinstance(output_dem, xr.core.dataarray.DataArray)
        assert not output_dem.dem.is_loaded
        assert output_dem.dem.raster_equal(trust_dem)
        assert isinstance(inlier_mask, xr.core.dataarray.DataArray)
        assert inlier_mask.rst.raster_equal(trust_mask)
    assert mask_path == xdem.examples.get_path("longyearbyen_glacier_outlines")


@pytest.mark.parametrize("chunks", [None, {"chunks": {"x": 200, "y": 200}}])
def test_load_dem_mask(tmp_path: Path, chunks: Dict[str, Any] | None) -> None:
    """
    Test load_dem function with different path_to_mask possibilities
    """

    # No path_to_mask
    config_dem = dict()
    config_dem["path_to_elev"] = "longyearbyen_ref_dem"
    output_dem, inlier_mask, mask_path = Workflows.load_dem(config_dem, chunks)
    assert mask_path is None
    assert inlier_mask is None

    # path_to_mask = shapefile path
    config_dem["path_to_mask"] = xdem.examples.get_path("longyearbyen_glacier_outlines")
    dem, inlier_mask, mask_path = Workflows.load_dem(config_dem, chunks)
    assert inlier_mask is not None
    reference_dem = xdem.DEM(xdem.examples.get_path(config_dem["path_to_elev"]))
    assert (
        inlier_mask.raster_equal(~gu.Vector(mask_path).create_mask(inlier_mask))
        if chunks is None
        else inlier_mask.rst.raster_equal(~gu.Vector(mask_path).create_mask(inlier_mask))
    )
    assert mask_path == config_dem["path_to_mask"]

    # path_to_mask = shapefile alias
    config_dem["path_to_mask"] = "longyearbyen_glacier_outlines"
    _, inlier_mask, mask_path = Workflows.load_dem(config_dem, chunks)
    assert (
        inlier_mask.raster_equal(~gu.Vector(mask_path).create_mask(inlier_mask))
        if chunks is None
        else inlier_mask.rst.raster_equal(~gu.Vector(mask_path).create_mask(inlier_mask))
    )
    assert mask_path == xdem.examples.get_path("longyearbyen_glacier_outlines")

    # path_to_mask = raster path, fitting dem
    mask = ~gu.Vector(mask_path).create_mask(reference_dem)
    mask.to_file(tmp_path / "mask.tif")
    config_dem["path_to_mask"] = str(tmp_path / "mask.tif")
    _, inlier_mask, mask_path = Workflows.load_dem(config_dem, chunks)
    assert inlier_mask is not None
    assert inlier_mask.raster_equal(mask) if chunks is None else inlier_mask.rst.raster_equal(mask)
    assert mask_path == config_dem["path_to_mask"]

    # path_to_mask = raster path, not fitting dem
    mask_crop = mask.icrop((0, 0, 500, 600))
    mask_crop.to_file(tmp_path / "mask_crop.tif")
    config_dem["path_to_mask"] = str(tmp_path / "mask_crop.tif")
    _, inlier_mask, mask_path = Workflows.load_dem(config_dem, chunks)
    assert inlier_mask is not None
    expected_mask = ~gu.Vector(tmp_path / "mask_crop.tif").create_mask(inlier_mask)
    assert inlier_mask.raster_equal(expected_mask) if chunks is None else inlier_mask.rst.raster_equal(expected_mask)
    assert mask_path == config_dem["path_to_mask"]

    # path_to_mask  = raster that not exists
    config_dem["path_to_mask"] = str(tmp_path / "notexist.tif")
    with pytest.raises(ValueError, match="not recognised as a mask"):
        Workflows.load_dem(config_dem, chunks)

    # path_to_mask = shapefile that not exists
    config_dem["path_to_mask"] = str(tmp_path / "notexist.shp")
    with pytest.raises(ValueError, match="not recognised as a mask"):
        Workflows.load_dem(config_dem, chunks)

    # path_to_mask = text file
    with open(tmp_path / "example.txt", "w") as f:
        f.write("some text")
    config_dem["path_to_mask"] = str(tmp_path / "example.txt")
    with pytest.raises(ValueError, match="not recognised as a mask"):
        Workflows.load_dem(config_dem, chunks)
