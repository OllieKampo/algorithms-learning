import math
from collections import deque

import matplotlib.pyplot as plt
import numpy as np

EXTEND_AREA = 1.0


def file_read(file_path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Read the lidar data from a file and return two np.arrays with the angles
    and distances respectively from the lidar sensor of each point.
    """
    with open(file_path) as data:
        measures = [line.split(",") for line in data]
    angles = []
    distances = []
    for measure in measures:
        angles.append(float(measure[0]))
        distances.append(float(measure[1]))
    return np.array(angles), np.array(distances)


def bresenham(start: tuple[int, int], end: tuple[int, int]) -> np.ndarray:
    """
    Implementation of Bresenham's line drawing algorithm.

    This algorithm is used to determine which points in a 2D grid should be
    plotted to form a line between two given points.

    Produces a np.array from start to end, for example:
    >>> points1 = bresenham((4, 4), (6, 10))
    >>> print(points1)
    np.array([[4,4], [4,5], [5,6], [5,7], [5,8], [6,9], [6,10]])
    """
    # setup initial conditions
    x1, y1 = start
    x2, y2 = end
    dx = x2 - x1
    dy = y2 - y1
    is_steep = abs(dy) > abs(dx)  # determine how steep the line is
    if is_steep:  # rotate line
        x1, y1 = y1, x1
        x2, y2 = y2, x2
    # swap start and end points if necessary and store swap state
    swapped = False
    if x1 > x2:
        x1, x2 = x2, x1
        y1, y2 = y2, y1
        swapped = True
    dx = x2 - x1  # recalculate differentials
    dy = y2 - y1  # recalculate differentials
    error = int(dx / 2.0)  # calculate error
    y_step = 1 if y1 < y2 else -1
    # iterate over bounding box generating points between start and end
    y = y1
    points = []
    for x in range(x1, x2 + 1):
        coord = (y, x) if is_steep else (x, y)
        points.append(coord)
        error -= abs(dy)
        if error < 0:
            y += y_step
            error += dx
    if swapped:  # reverse the list if the coordinates were swapped
        points.reverse()
    return np.array(points)


def calc_grid_map_config(
    points_x: np.ndarray,
    points_y: np.ndarray,
    xy_resolution: float
) -> tuple[int, int, int, int, int, int]:
    """
    Calculates the size and the maximum distances of the grid map.

    points_x: The x coordinates of the lidar data in Cartesian coordinates.
    points_y: The y coordinates of the lidar data in Cartesian coordinates.
    xy_resolution: The resolution of the grid map.

    Returns:
    The minimum x, minimum y, maximum x, maximum y, x width in grid cells, and
    y width in grid cells.
    """
    min_x = round(min(points_x) - EXTEND_AREA / 2.0)
    min_y = round(min(points_y) - EXTEND_AREA / 2.0)
    max_x = round(max(points_x) + EXTEND_AREA / 2.0)
    max_y = round(max(points_y) + EXTEND_AREA / 2.0)
    xw = int(round((max_x - min_x) / xy_resolution))
    yw = int(round((max_y - min_y) / xy_resolution))
    return min_x, min_y, max_x, max_y, xw, yw


def atan_zero_to_twopi(y, x):
    angle = math.atan2(y, x)
    if angle < 0.0:
        angle += math.pi * 2.0
    return angle


def flood_fill(center_point, occupancy_map):
    """
    center_point: starting point (x,y) of fill
    occupancy_map: occupancy map generated from Bresenham ray-tracing
    """
    # Fill empty areas with queue method
    sx, sy = occupancy_map.shape
    fringe = deque()
    fringe.appendleft(center_point)
    while fringe:
        n = fringe.pop()
        nx, ny = n
        # West
        if nx > 0:
            if occupancy_map[nx - 1, ny] == 0.5:
                occupancy_map[nx - 1, ny] = 0.0
                fringe.appendleft((nx - 1, ny))
        # East
        if nx < sx - 1:
            if occupancy_map[nx + 1, ny] == 0.5:
                occupancy_map[nx + 1, ny] = 0.0
                fringe.appendleft((nx + 1, ny))
        # North
        if ny > 0:
            if occupancy_map[nx, ny - 1] == 0.5:
                occupancy_map[nx, ny - 1] = 0.0
                fringe.appendleft((nx, ny - 1))
        # South
        if ny < sy - 1:
            if occupancy_map[nx, ny + 1] == 0.5:
                occupancy_map[nx, ny + 1] = 0.0
                fringe.appendleft((nx, ny + 1))


def generate_ray_casting_grid_map(
    points_x: np.ndarray,
    points_y: np.ndarray,
    xy_resolution: float,
    breshen: bool = True
) -> np.ndarray:
    """
    Generate a grid map from the lidar data using ray casting.

    points_x: The x coordinates of the lidar data in Cartesian coordinates.
    points_y: The y coordinates of the lidar data in Cartesian coordinates.
    xy_resolution: The resolution of the grid map.
    breshen: If True, use Bresenham ray casting. If False, use flood fill.

    Returns:
    A grid map of the environment.
    """
    min_x, min_y, max_x, max_y, x_w, y_w = calc_grid_map_config(
        points_x, points_y, xy_resolution
    )

    # default 0.5 -- [[0.5 for i in range(y_w)] for i in range(x_w)]
    occupancy_map = np.ones((x_w, y_w)) * 0.5
    center_x = int(
        round(-min_x / xy_resolution))  # center x coordinate of the grid map
    center_y = int(
        round(-min_y / xy_resolution))  # center y coordinate of the grid map

    # occupancy grid computed with bresenham ray casting
    if breshen:
        for (x, y) in zip(points_x, points_y):
            # x coordinate of the the occupied area
            ix = int(round((x - min_x) / xy_resolution))
            # y coordinate of the the occupied area
            iy = int(round((y - min_y) / xy_resolution))
            laser_beams = bresenham((center_x, center_y), (
                ix, iy))  # line form the lidar to the occupied point
            for laser_beam in laser_beams:
                occupancy_map[laser_beam[0]][
                    laser_beam[1]] = 0.0  # free area 0.0
            occupancy_map[ix][iy] = 1.0  # occupied area 1.0
            occupancy_map[ix + 1][iy] = 1.0  # extend the occupied area
            occupancy_map[ix][iy + 1] = 1.0  # extend the occupied area
            occupancy_map[ix + 1][iy + 1] = 1.0  # extend the occupied area

    # occupancy grid computed with with flood fill
    else:
        prev_ix, prev_iy = points_x[0], points_y[0]
        ix = prev_ix = int(round((prev_ix - min_x) / xy_resolution))
        iy = prev_iy = int(round((prev_iy - min_y) / xy_resolution))
        occupancy_map[ix][iy] = 1.0  # occupied area 1.0
        occupancy_map[ix + 1][iy] = 1.0  # extend the occupied area
        occupancy_map[ix][iy + 1] = 1.0  # extend the occupied area
        occupancy_map[ix + 1][iy + 1] = 1.0  # extend the occupied area
        for (x, y) in zip(points_x[1:], points_y[1:]):
            # x coordinate of the occupied area
            ix = int(round((x - min_x) / xy_resolution))
            # y coordinate of the occupied area
            iy = int(round((y - min_y) / xy_resolution))
            ray_between_obstacles = bresenham((prev_ix, prev_iy), (ix, iy))
            for obstacle in ray_between_obstacles:
                if occupancy_map[obstacle[0]][obstacle[1]] != 1.0:
                    occupancy_map[obstacle[0]][obstacle[1]] = 0.0  # obstacle area 1.0
            occupancy_map[ix][iy] = 1.0  # occupied area 1.0
            occupancy_map[ix + 1][iy] = 1.0  # extend the occupied area
            occupancy_map[ix][iy + 1] = 1.0  # extend the occupied area
            occupancy_map[ix + 1][iy + 1] = 1.0  # extend the occupied area
            prev_ix = ix
            prev_iy = iy

        flood_fill((center_x, center_y), occupancy_map)

    return occupancy_map


def main():
    xy_resolution = 0.02  # x-y grid resolution

    # Load the lidar data as polar coordinates.
    ang, dist = file_read("lidar01.csv")

    # Convert the polar coordinates to Cartesian coordinates.
    points_x = np.sin(ang) * dist
    points_y = np.cos(ang) * dist

    occupancy_map = generate_ray_casting_grid_map(
        points_x,
        points_y,
        xy_resolution,
        breshen=False
    )

    xy_res = np.array(occupancy_map).shape
    plt.figure(1, figsize=(10, 4))
    plt.subplot(122)
    plt.imshow(occupancy_map, cmap="PiYG_r")
    # cmap = "binary" "PiYG_r" "PiYG_r" "bone" "bone_r" "RdYlGn_r"
    plt.clim(-0.4, 1.4)
    plt.gca().set_xticks(np.arange(-.5, xy_res[1], 1), minor=True)
    plt.gca().set_yticks(np.arange(-.5, xy_res[0], 1), minor=True)
    plt.grid(True, which="minor", color="w", linewidth=0.6, alpha=0.5)
    plt.colorbar()
    plt.subplot(121)
    plt.plot([points_y, np.zeros(np.size(points_y))], [points_x, np.zeros(np.size(points_y))], "ro-")
    plt.axis("equal")
    plt.plot(0.0, 0.0, "ob")
    plt.gca().set_aspect("equal", "box")
    bottom, top = plt.ylim()  # return the current y-lim
    plt.ylim((top, bottom))  # rescale y axis, to match the grid orientation
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    main()
