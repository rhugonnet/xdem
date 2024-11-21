# Copyright (c) 2024 xDEM developers
#
# This file is part of xDEM project:
# https://github.com/glaciohack/xdem
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from xdem import (  # noqa
    coreg,
    dem,
    examples,
    filters,
    fit,
    spatialstats,
    terrain,
    volume,
)
from xdem.ddem import dDEM  # noqa
from xdem.dem import DEM, xr_accessor  # noqa
from xdem.dem.xr_accessor import open_dem  # noqa
from xdem.demcollection import DEMCollection  # noqa

try:
    from xdem._version import __version__  # noqa
except ImportError:  # pragma: no cover
    raise ImportError(
        "xDEM is not properly installed. If you are "
        "running from the source directory, please instead "
        "create a new virtual environment (using conda or "
        "virtualenv) and then install it in-place by running: "
        "pip install -e ."
    )
