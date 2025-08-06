# Copyright (c) 2025, The Nav-Suite Project Developers (https://github.com/leggedrobotics/nav-suite/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Package to import terrains for navigation tasks."""

import os
import toml

# Conveniences to other module directories via relative paths
NAVSUITE_EXT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
"""Path to the extension source directory."""

NAVSUITE_DATA_DIR = os.path.join(NAVSUITE_EXT_DIR, "data")
"""Path to the extension data directory."""

NAVSUITE_TEST_ASSETS_DIR = os.path.join(NAVSUITE_EXT_DIR, "tests", "assets")
"""Path to the extension test assets directory."""

NAVSUITE_METADATA = toml.load(os.path.join(NAVSUITE_EXT_DIR, "config", "extension.toml"))
"""Extension metadata dictionary parsed from the extension.toml file."""

# Configure the module-level variables
__version__ = NAVSUITE_METADATA["package"]["version"]
