# Copyright (c) 2025, The Nav-Suite Project Developers (https://github.com/leggedrobotics/nav-suite/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Package for navigation tasks."""

import os
import toml

# Conveniences to other module directories via relative paths
NAVSUITE_TASKS_EXT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
"""Path to the extension source directory."""

NAVSUITE_TASKS_DATA_DIR = os.path.join(NAVSUITE_TASKS_EXT_DIR, "data")
"""Path to the extension data directory."""

NAVSUITE_TASKS_METADATA = toml.load(os.path.join(NAVSUITE_TASKS_EXT_DIR, "config", "extension.toml"))
"""Extension metadata dictionary parsed from the extension.toml file."""

# Configure the module-level variables
__version__ = NAVSUITE_TASKS_METADATA["package"]["version"]
