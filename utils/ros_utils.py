import numpy as np

def generate_occupancy_map_from_point_cloud(
    point_cloud: np.ndarray,
    map_size: Tuple[int, int] = (60, 60),
    resolution: float = 0.1,
) -> np.ndarray:
    """
    Generate an occupancy map from a laser scan.

    Args:
        point_cloud: Point cloud of shape: (num_points, 2 or 3)
        map_size: The size of the occupancy map in pixels.
        resolution: The resolution of the occupancy map in meters per pixel.

    Returns:
        The occupancy map with shape (map_size[0], map_size[1]).
    """
    map_height, map_width = map_size

    origin = (map_width * resolution / 2, map_height * resolution / 2)

    # Polar to Cartesian coordinates transformation
    x = point_cloud[:, 0]
    y = point_cloud[:, 1]

    # Transformation to map frame
    map_x = np.round((x + origin[0]) / resolution).astype(int)
    map_y = np.round((y + origin[1]) / resolution).astype(int)

    valid_points = (map_x >= 0) & (map_x < map_width) & (map_y >= 0) & (map_y < map_height)
    map_x = map_x[valid_points]
    map_y = map_y[valid_points]

    occupancy_map = np.zeros(map_size, dtype=np.int8)

    if len(map_x) > 0:
        occupancy_map[map_y, map_x] = 100  # Occupied

    # PLACEHOLDER ego-agent position on map
    ego_x = int(map_width / 2)
    ego_y = int(map_height / 2)
    occupancy_map[ego_y, ego_x] = 50  # Ego-agent

    return occupancy_map

