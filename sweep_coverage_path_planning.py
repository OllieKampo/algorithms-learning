import math
from enum import IntEnum
from functools import total_ordering

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as Rot

do_animation = True


def rot_mat_2d(angle):
    """
    Create 2D rotation matrix from an angle

    Parameters
    ----------
    angle :

    Returns
    -------
    A 2D rotation matrix

    Examples
    --------
    >>> angle_mod(-4.0)


    """
    return Rot.from_euler('z', angle).as_matrix()[0:2, 0:2]


@total_ordering
class FloatGrid:

    def __init__(self, init_val=0.0):
        self.data = init_val

    def get_float_data(self):
        return self.data

    def __eq__(self, other):
        if not isinstance(other, FloatGrid):
            return NotImplemented
        return self.get_float_data() == other.get_float_data()

    def __lt__(self, other):
        if not isinstance(other, FloatGrid):
            return NotImplemented
        return self.get_float_data() < other.get_float_data()


class GridMap:
    """
    GridMap class
    """

    def __init__(self, width, height, resolution,
                 center_x, center_y, init_val=FloatGrid(0.0)):
        """__init__

        :param width: number of grid for width
        :param height: number of grid for height
        :param resolution: grid resolution [m]
        :param center_x: center x position  [m]
        :param center_y: center y position [m]
        :param init_val: initial value for all grid
        """
        self.width = width
        self.height = height
        self.resolution = resolution
        self.center_x = center_x
        self.center_y = center_y

        self.left_lower_x = self.center_x - self.width / 2.0 * self.resolution
        self.left_lower_y = self.center_y - self.height / 2.0 * self.resolution

        self.n_data = self.width * self.height
        self.data = [init_val] * self.n_data
        self.data_type = type(init_val)

    def get_value_from_xy_index(self, x_ind, y_ind):
        """get_value_from_xy_index

        when the index is out of grid map area, return None

        :param x_ind: x index
        :param y_ind: y index
        """

        grid_ind = self.calc_grid_index_from_xy_index(x_ind, y_ind)

        if 0 <= grid_ind < self.n_data:
            return self.data[grid_ind]
        else:
            return None

    def get_xy_index_from_xy_pos(self, x_pos, y_pos):
        """get_xy_index_from_xy_pos

        :param x_pos: x position [m]
        :param y_pos: y position [m]
        """
        x_ind = self.calc_xy_index_from_position(
            x_pos, self.left_lower_x, self.width)
        y_ind = self.calc_xy_index_from_position(
            y_pos, self.left_lower_y, self.height)

        return x_ind, y_ind

    def set_value_from_xy_pos(self, x_pos, y_pos, val):
        """set_value_from_xy_pos

        return bool flag, which means setting value is succeeded or not

        :param x_pos: x position [m]
        :param y_pos: y position [m]
        :param val: grid value
        """

        x_ind, y_ind = self.get_xy_index_from_xy_pos(x_pos, y_pos)

        if (not x_ind) or (not y_ind):
            return False  # NG

        flag = self.set_value_from_xy_index(x_ind, y_ind, val)

        return flag

    def set_value_from_xy_index(self, x_ind, y_ind, val):
        """set_value_from_xy_index

        return bool flag, which means setting value is succeeded or not

        :param x_ind: x index
        :param y_ind: y index
        :param val: grid value
        """

        if (x_ind is None) or (y_ind is None):
            return False, False

        grid_ind = int(y_ind * self.width + x_ind)

        if 0 <= grid_ind < self.n_data and isinstance(val, self.data_type):
            self.data[grid_ind] = val
            return True  # OK
        else:
            return False  # NG

    def set_value_from_polygon(self, pol_x, pol_y, val, inside=True):
        """set_value_from_polygon

        Setting value inside or outside polygon

        :param pol_x: x position list for a polygon
        :param pol_y: y position list for a polygon
        :param val: grid value
        :param inside: setting data inside or outside
        """

        # making ring polygon
        if (pol_x[0] != pol_x[-1]) or (pol_y[0] != pol_y[-1]):
            np.append(pol_x, pol_x[0])
            np.append(pol_y, pol_y[0])

        # setting value for all grid
        for x_ind in range(self.width):
            for y_ind in range(self.height):
                x_pos, y_pos = self.calc_grid_central_xy_position_from_xy_index(
                    x_ind, y_ind)

                flag = self.check_inside_polygon(x_pos, y_pos, pol_x, pol_y)

                if flag is inside:
                    self.set_value_from_xy_index(x_ind, y_ind, val)

    def calc_grid_index_from_xy_index(self, x_ind, y_ind):
        grid_ind = int(y_ind * self.width + x_ind)
        return grid_ind

    def calc_xy_index_from_grid_index(self, grid_ind):
        y_ind, x_ind = divmod(grid_ind, self.width)
        return x_ind, y_ind

    def calc_grid_index_from_xy_pos(self, x_pos, y_pos):
        """get_xy_index_from_xy_pos

        :param x_pos: x position [m]
        :param y_pos: y position [m]
        """
        x_ind = self.calc_xy_index_from_position(
            x_pos, self.left_lower_x, self.width)
        y_ind = self.calc_xy_index_from_position(
            y_pos, self.left_lower_y, self.height)

        return self.calc_grid_index_from_xy_index(x_ind, y_ind)

    def calc_grid_central_xy_position_from_grid_index(self, grid_ind):
        x_ind, y_ind = self.calc_xy_index_from_grid_index(grid_ind)
        return self.calc_grid_central_xy_position_from_xy_index(x_ind, y_ind)

    def calc_grid_central_xy_position_from_xy_index(self, x_ind, y_ind):
        x_pos = self.calc_grid_central_xy_position_from_index(
            x_ind, self.left_lower_x)
        y_pos = self.calc_grid_central_xy_position_from_index(
            y_ind, self.left_lower_y)

        return x_pos, y_pos

    def calc_grid_central_xy_position_from_index(self, index, lower_pos):
        return lower_pos + index * self.resolution + self.resolution / 2.0

    def calc_xy_index_from_position(self, pos, lower_pos, max_index):
        ind = int(np.floor((pos - lower_pos) / self.resolution))
        if 0 <= ind <= max_index:
            return ind
        else:
            return None

    def check_occupied_from_xy_index(self, x_ind, y_ind, occupied_val):

        val = self.get_value_from_xy_index(x_ind, y_ind)

        if val is None or val >= occupied_val:
            return True
        else:
            return False

    def expand_grid(self, occupied_val=FloatGrid(1.0)):
        x_inds, y_inds, values = [], [], []

        for ix in range(self.width):
            for iy in range(self.height):
                if self.check_occupied_from_xy_index(ix, iy, occupied_val):
                    x_inds.append(ix)
                    y_inds.append(iy)
                    values.append(self.get_value_from_xy_index(ix, iy))

        for (ix, iy, value) in zip(x_inds, y_inds, values):
            self.set_value_from_xy_index(ix + 1, iy, val=value)
            self.set_value_from_xy_index(ix, iy + 1, val=value)
            self.set_value_from_xy_index(ix + 1, iy + 1, val=value)
            self.set_value_from_xy_index(ix - 1, iy, val=value)
            self.set_value_from_xy_index(ix, iy - 1, val=value)
            self.set_value_from_xy_index(ix - 1, iy - 1, val=value)

    @staticmethod
    def check_inside_polygon(iox, ioy, x, y):

        n_point = len(x) - 1
        inside = False
        for i1 in range(n_point):
            i2 = (i1 + 1) % (n_point + 1)

            if x[i1] >= x[i2]:
                min_x, max_x = x[i2], x[i1]
            else:
                min_x, max_x = x[i1], x[i2]
            if not min_x <= iox < max_x:
                continue

            tmp1 = (y[i2] - y[i1]) / (x[i2] - x[i1])
            if (y[i1] + tmp1 * (iox - x[i1]) - ioy) > 0.0:
                inside = not inside

        return inside

    def print_grid_map_info(self):
        print("width:", self.width)
        print("height:", self.height)
        print("resolution:", self.resolution)
        print("center_x:", self.center_x)
        print("center_y:", self.center_y)
        print("left_lower_x:", self.left_lower_x)
        print("left_lower_y:", self.left_lower_y)
        print("n_data:", self.n_data)

    def plot_grid_map(self, ax=None):
        float_data_array = np.array([d.get_float_data() for d in self.data])
        grid_data = np.reshape(float_data_array, (self.height, self.width))
        if not ax:
            fig, ax = plt.subplots()
        heat_map = ax.pcolor(grid_data, cmap="Blues", vmin=0.0, vmax=1.0)
        plt.axis("equal")

        return heat_map


def check_occupied(c_x_index, c_y_index, grid_map, occupied_val=FloatGrid(0.5)):
    return grid_map.check_occupied_from_xy_index(c_x_index, c_y_index, occupied_val)


class SweepSearcher:
    class SweepDirection(IntEnum):
        UP = 1
        DOWN = -1

    class MovingDirection(IntEnum):
        RIGHT = 1
        LEFT = -1

    def __init__(
        self,
        moving_direction,
        sweep_direction,
        x_inds_goal_y,
        goal_y
    ) -> None:
        self.moving_direction = moving_direction
        self.sweep_direction = sweep_direction
        self.turning_window: list[tuple[float, float]] = []
        self.update_turning_window()
        self.x_indexes_goal_y = x_inds_goal_y
        self.goal_y = goal_y

    def move_target_grid(self, c_x_index, c_y_index, grid_map):
        n_x_index = self.moving_direction + c_x_index
        n_y_index = c_y_index

        # found safe grid
        if not check_occupied(n_x_index, n_y_index, grid_map):
            return n_x_index, n_y_index
        else:  # occupied
            next_c_x_index, next_c_y_index = self.find_safe_turning_grid(
                c_x_index, c_y_index, grid_map)
            if (next_c_x_index is None) and (next_c_y_index is None):
                # moving backward
                next_c_x_index = -self.moving_direction + c_x_index
                next_c_y_index = c_y_index
                if check_occupied(next_c_x_index, next_c_y_index, grid_map, FloatGrid(1.0)):
                    # moved backward, but the grid is occupied by obstacle
                    return None, None
            else:
                # keep moving until end
                while not check_occupied(next_c_x_index + self.moving_direction, next_c_y_index, grid_map):
                    next_c_x_index += self.moving_direction
                self.swap_moving_direction()
            return next_c_x_index, next_c_y_index

    def find_safe_turning_grid(self, c_x_index, c_y_index, grid_map):

        for (d_x_ind, d_y_ind) in self.turning_window:

            next_x_ind = d_x_ind + c_x_index
            next_y_ind = d_y_ind + c_y_index

            # found safe grid
            if not check_occupied(next_x_ind, next_y_ind, grid_map):
                return next_x_ind, next_y_ind

        return None, None

    def is_search_done(self, grid_map):
        for ix in self.x_indexes_goal_y:
            if not check_occupied(ix, self.goal_y, grid_map):
                return False

        # all lower grid is occupied
        return True

    def update_turning_window(self):
        # turning window definition
        # robot can move grid based on it.
        self.turning_window = [
            (self.moving_direction, 0.0),
            (self.moving_direction, self.sweep_direction),
            (0, self.sweep_direction),
            (-self.moving_direction, self.sweep_direction),
        ]

    def swap_moving_direction(self):
        self.moving_direction *= -1
        self.update_turning_window()

    def search_start_grid(self, grid_map):
        x_inds = []
        y_ind = 0
        if self.sweep_direction == self.SweepDirection.DOWN:
            x_inds, y_ind = search_free_grid_index_at_edge_y(
                grid_map, from_upper=True)
        elif self.sweep_direction == self.SweepDirection.UP:
            x_inds, y_ind = search_free_grid_index_at_edge_y(
                grid_map, from_upper=False)

        if self.moving_direction == self.MovingDirection.RIGHT:
            return min(x_inds), y_ind
        elif self.moving_direction == self.MovingDirection.LEFT:
            return max(x_inds), y_ind

        raise ValueError("self.moving direction is invalid ")


def find_sweep_direction_and_start_position(
    polygon_boundries_x: list[float],
    polygon_boundries_y: list[float]
) -> tuple[list[float], list[float]]:
    """
    This function could be used in a path planning algorithm to determine the
    direction and starting position of a sweep over a polygonal area.

    The function takes two lists as input: polygon_boundries_x and polygon_boundries_y.
    These lists represent the x and y coordinates of the vertices of a polygon, respectively.

    The function returns a tuple of two lists. The first list, vec, represents the
    direction of the longest edge of the polygon. The second list, sweep_start_pos,
    represents the starting position of the sweep, which is the first vertex of the longest edge.
    """

    # Initialize max_dist (the maximum distance, initially set to 0),
    # vec (the vector representing the direction of the longest edge, initially set to [0.0, 0.0]),
    # and sweep_start_pos (the starting position of the sweep, initially set to [0.0, 0.0]).
    max_dist: float = 0.0
    sweep_vec: list[float] = [0.0, 0.0]
    sweep_start_pos: list[float] = [0.0, 0.0]

    # It then iterates over the vertices of the polygon. For each pair of consecutive vertices,
    # it calculates the distance between them using the Pythagorean theorem (np.hypot(dx, dy)).
    for i in range(len(polygon_boundries_x) - 1):
        dx = polygon_boundries_x[i + 1] - polygon_boundries_x[i]
        dy = polygon_boundries_y[i + 1] - polygon_boundries_y[i]
        distance = np.hypot(dx, dy)

        # If the calculated distance is greater than the current max_dist, it updates
        # max_dist with the new distance, vec with the vector formed by the two vertices,
        # and sweep_start_pos with the coordinates of the first vertex.
        if distance > max_dist:
            max_dist = distance
            sweep_vec = [dx, dy]
            sweep_start_pos = [polygon_boundries_x[i], polygon_boundries_y[i]]

    # Return the calculated sweep vector and starting position.
    return sweep_vec, sweep_start_pos


def convert_grid_coordinate(
    polygon_boundries_x: list[float],
    polygon_boundries_y: list[float],
    sweep_vec: list[float],
    sweep_start_position: list[float]
) -> tuple[list[float], list[float]]:
    """
    This function is used to convert the coordinates of the vertices of the
    polygon from the original coordinate system to a new coordinate system
    where the x-axis is aligned with the sweep vector and the origin is at the
    starting position of the sweep. This is typically done in path planning
    algorithms to simplify the calculation of the sweep path.

    The function returns a tuple of two lists, which represent the x and y
    coordinates of the vertices of the polygon in the new coordinate system.
    """
    # Calculate the relative x and y coordinates of the vertices of the polygon with respect
    # to the starting position of the sweep.
    relative_polygon_boundries_x = [ix - sweep_start_position[0] for ix in polygon_boundries_x]
    relative_polygon_boundries_y = [iy - sweep_start_position[1] for iy in polygon_boundries_y]

    # Calculate the angle (theta) of the sweep vector with respect to the x-axis origin and
    # rotate the relative coordinates of the polygon vertices by this angle to align the x-axis
    # with the sweep vector.
    theta = math.atan2(sweep_vec[1], sweep_vec[0])
    polygon_vertices = np.stack([relative_polygon_boundries_x, relative_polygon_boundries_y]).T
    converted_polygon_vertices = polygon_vertices @ rot_mat_2d(theta)

    # Return the x and y coordinates of the vertices of the polygon in the new coordinate system.
    return converted_polygon_vertices[:, 0], converted_polygon_vertices[:, 1]


def convert_global_coordinate(x, y, sweep_vec, sweep_start_position):
    th = math.atan2(sweep_vec[1], sweep_vec[0])
    converted_xy = np.stack([x, y]).T @ rot_mat_2d(-th)
    rx = [ix + sweep_start_position[0] for ix in converted_xy[:, 0]]
    ry = [iy + sweep_start_position[1] for iy in converted_xy[:, 1]]
    return rx, ry


def search_free_grid_index_at_edge_y(
    grid_map: GridMap,
    from_upper: bool
) -> tuple[list[int], int | None]:
    """
    This function could be used in a path planning algorithm to find a free
    cell on the edge of a grid map to start or end a path.
    """
    # It initializes y_index (the y-coordinate of the edge) to None and
    # x_indexes (the list of x-coordinates of free cells) to an empty list.
    y_index: int | None = None
    x_indexes: list[float] = []

    # It sets the ranges of x and y coordinates for the search.
    # If from_upper is True, it searches from the upper edge of the grid map
    # (the ranges are reversed). Otherwise, it searches from the lower edge.
    x_range = range(grid_map.width)
    y_range = range(grid_map.height)
    if from_upper:
        x_range = x_range[::-1]
        y_range = y_range[::-1]

    # It iterates over the cells of the grid map in the specified ranges.
    # For each cell, it checks if the cell is not occupied.
    for iy in y_range:
        # Find all columns that are not occupied in this row.
        for ix in x_range:
            if not check_occupied(ix, iy, grid_map):
                y_index = iy
                x_indexes.append(ix)
        if y_index:
            break

    # Return all free x indices for first free y index.
    return x_indexes, y_index


def setup_grid_map(ox, oy, resolution, sweep_direction, offset_grid=10):
    width = math.ceil((max(ox) - min(ox)) / resolution) + offset_grid
    height = math.ceil((max(oy) - min(oy)) / resolution) + offset_grid
    center_x = (np.max(ox) + np.min(ox)) / 2.0
    center_y = (np.max(oy) + np.min(oy)) / 2.0

    grid_map = GridMap(width, height, resolution, center_x, center_y)
    grid_map.print_grid_map_info()
    grid_map.set_value_from_polygon(ox, oy, FloatGrid(1.0), inside=False)
    grid_map.expand_grid()

    x_inds_goal_y: list[int] = []
    goal_y: int = 0
    if sweep_direction == SweepSearcher.SweepDirection.UP:
        x_inds_goal_y, goal_y = search_free_grid_index_at_edge_y(
            grid_map,
            from_upper=True
        )
    elif sweep_direction == SweepSearcher.SweepDirection.DOWN:
        x_inds_goal_y, goal_y = search_free_grid_index_at_edge_y(
            grid_map,
            from_upper=False
        )

    return grid_map, x_inds_goal_y, goal_y


def sweep_path_search(
    sweep_searcher: SweepSearcher,
    grid_map: GridMap
) -> tuple[list[float], list[float]]:
    # search start grid
    c_x_index, c_y_index = sweep_searcher.search_start_grid(grid_map)
    if not grid_map.set_value_from_xy_index(c_x_index, c_y_index, FloatGrid(0.5)):
        print("Cannot find start grid")
        return [], []

    x, y = grid_map.calc_grid_central_xy_position_from_xy_index(c_x_index,
                                                                c_y_index)
    px, py = [x], [y]

    while True:
        c_x_index, c_y_index = sweep_searcher.move_target_grid(c_x_index,
                                                               c_y_index,
                                                               grid_map)

        if sweep_searcher.is_search_done(grid_map) or (
                c_x_index is None or c_y_index is None):
            break

        x, y = grid_map.calc_grid_central_xy_position_from_xy_index(
            c_x_index, c_y_index)

        px.append(x)
        py.append(y)

        grid_map.set_value_from_xy_index(c_x_index, c_y_index, FloatGrid(0.5))

    return px, py


def planning(
    polygon_boundries_x: list[float],
    polygon_boundries_y: list[float],
    resolution: float,
    moving_direction: SweepSearcher.MovingDirection = SweepSearcher.MovingDirection.RIGHT,
    sweeping_direction: SweepSearcher.SweepDirection = SweepSearcher.SweepDirection.UP,
):
    """
    This is the actual planning algorithm itself.
    """
    sweep_vec, sweep_start_position = find_sweep_direction_and_start_position(
        polygon_boundries_x,
        polygon_boundries_y
    )

    translated_polygon_boundries_x, translated_polygon_boundries_y = convert_grid_coordinate(
        polygon_boundries_x,
        polygon_boundries_y,
        sweep_vec,
        sweep_start_position
    )

    grid_map, x_inds_goal_y, goal_y = setup_grid_map(
        translated_polygon_boundries_x,
        translated_polygon_boundries_y,
        resolution,
        sweeping_direction
    )

    sweep_searcher = SweepSearcher(
        moving_direction,
        sweeping_direction,
        x_inds_goal_y,
        goal_y
    )

    px, py = sweep_path_search(sweep_searcher, grid_map)

    rx, ry = convert_global_coordinate(px, py, sweep_vec,
                                       sweep_start_position)

    print("Path length:", len(rx))

    return rx, ry


def planning_animation(ox, oy, resolution):  # pragma: no cover
    px, py = planning(ox, oy, resolution)

    # animation
    if do_animation:
        for ipx, ipy in zip(px, py):
            plt.cla()
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect(
                'key_release_event',
                lambda event: [exit(0) if event.key == 'escape' else None])
            plt.plot(ox, oy, "-xb")
            plt.plot(px, py, "-r")
            plt.plot(ipx, ipy, "or")
            plt.axis("equal")
            plt.grid(True)
            plt.pause(0.1)

        plt.cla()
        plt.plot(ox, oy, "-xb")
        plt.plot(px, py, "-r")
        plt.axis("equal")
        plt.grid(True)
        plt.pause(0.1)
        plt.close()


def main():  # pragma: no cover
    print("start!!")

    ox = [0.0, 20.0, 50.0, 100.0, 130.0, 40.0, 0.0]
    oy = [0.0, -20.0, 0.0, 30.0, 60.0, 80.0, 0.0]
    resolution = 5.0
    planning_animation(ox, oy, resolution)

    ox = [0.0, 50.0, 50.0, 0.0, 0.0]
    oy = [0.0, 0.0, 30.0, 30.0, 0.0]
    resolution = 1.3
    planning_animation(ox, oy, resolution)

    ox = [0.0, 20.0, 50.0, 200.0, 130.0, 40.0, 0.0]
    oy = [0.0, -80.0, 0.0, 30.0, 60.0, 80.0, 0.0]
    resolution = 5.0
    planning_animation(ox, oy, resolution)

    if do_animation:
        plt.show()
    print("done!!")


if __name__ == '__main__':
    main()