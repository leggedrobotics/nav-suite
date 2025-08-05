# Copyright (c) 2025, The Nav-Suite Project Developers (https://github.com/leggedrobotics/nav-suite/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0


def lin_interp(start_step: int, end_step: int, start_val: float, end_val: float, current_step: int) -> float:
    """Linearly interpolate a parameter between two values.

    Args:
        start_step: The step at which the parameter starts.
        end_step: The step at which the parameter ends.
        start_val: The parameter value at the start.
        end_val: The parameter value at the end.
        current_step: The current step.

    Returns:
        The interpolated parameter value.
    """
    num_steps = end_step - start_step
    new_value = (
        start_val
        + (min(end_step - start_step, max(0.0, current_step - start_step))) * (end_val - start_val) / num_steps
    )

    return new_value
