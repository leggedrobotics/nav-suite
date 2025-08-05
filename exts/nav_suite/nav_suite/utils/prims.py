# Copyright (c) 2025, The Nav-Suite Project Developers (https://github.com/leggedrobotics/nav-suite/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import isaacsim.core.utils.prims as prim_utils
from pxr import Usd


def get_all_prims_including_str(start_prim: str, path: str) -> list[Usd.Prim]:
    """Get all prims that include the given path str.

    This function recursively searches for all prims that include the given path str.

    Args:
        start_prim: The environment prim path from which to begin the search.
        path: The path string to search for.

    Returns:
        A list of all prims that include the given path str.
    """

    def recursive_search(start_prim: Usd.Prim, prim_name: str, found_prims: list) -> list[Usd.Prim]:
        for curr_prim in start_prim.GetChildren():
            if prim_name.lower() in curr_prim.GetPath().pathString.lower():
                found_prims.append(curr_prim)
            else:
                found_prims = recursive_search(start_prim=curr_prim, prim_name=prim_name, found_prims=found_prims)

        return found_prims

    # Raise error if the start prim is not valid
    assert prim_utils.is_prim_path_valid(start_prim), f"Prim path '{start_prim}' is not valid"

    start_prim = prim_utils.get_prim_at_path(start_prim)
    final_prim = recursive_search(start_prim=start_prim, prim_name=path, found_prims=[])
    return final_prim
