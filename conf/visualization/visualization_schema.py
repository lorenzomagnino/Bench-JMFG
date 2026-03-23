from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ColorsConfig:
    """Color configuration for visualization."""

    mean_field_1d_bar: str = "salmon"
    mean_field_1d_background: str = "floralwhite"
    mean_field_1d_grid: str = "white"

    mean_field_2d_cmap: str = "viridis"
    mean_field_2d_walls_cmap: str = "Greys"
    mean_field_2d_grid: str = "lightgray"

    mean_field_3d_cmap: str = "viridis"

    policy_cmap: str = "viridis"
    policy_grid: str = "gray"

    figure_background: str = "white"

    policy2d_action_cmaps: Optional[list[str]] = field(
        default_factory=lambda: ["Greens", "Purples", "Oranges", "Blues", "Greys"]
    )
    policy2d_action_labels: Optional[list[str]] = field(
        default_factory=lambda: ["up", "right", "down", "left", "stay"]
    )
