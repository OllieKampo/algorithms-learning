import itertools
from enum import Enum

import matplotlib.pyplot as plt
import numpy as np

import math
import random

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation


def rot_mat_2d(angle):
    return Rotation.from_euler('z', angle).as_matrix()[0:2, 0:2]


class VehicleSimulator:

    def __init__(self, i_x, i_y, i_yaw, i_v, max_v, w, L):
        self.x = i_x
        self.y = i_y
        self.yaw = i_yaw
        self.v = i_v
        self.max_v = max_v
        self.W = w
        self.L = L
        self._calc_vehicle_contour()

    def update(self, dt, a, omega):
        self.x += self.v * np.cos(self.yaw) * dt
        self.y += self.v * np.sin(self.yaw) * dt
        self.yaw += omega * dt
        self.v += a * dt
        if self.v >= self.max_v:
            self.v = self.max_v

    def plot(self):
        plt.plot(self.x, self.y, ".b")

        # convert global coordinate
        gx, gy = self.calc_global_contour()
        plt.plot(gx, gy, "--b")

    def calc_global_contour(self):
        gxy = np.stack([self.vc_x, self.vc_y]).T @ rot_mat_2d(self.yaw)
        gx = gxy[:, 0] + self.x
        gy = gxy[:, 1] + self.y

        return gx, gy

    def _calc_vehicle_contour(self):

        self.vc_x = []
        self.vc_y = []

        self.vc_x.append(self.L / 2.0)
        self.vc_y.append(self.W / 2.0)

        self.vc_x.append(self.L / 2.0)
        self.vc_y.append(-self.W / 2.0)

        self.vc_x.append(-self.L / 2.0)
        self.vc_y.append(-self.W / 2.0)

        self.vc_x.append(-self.L / 2.0)
        self.vc_y.append(self.W / 2.0)

        self.vc_x.append(self.L / 2.0)
        self.vc_y.append(self.W / 2.0)

        self.vc_x, self.vc_y = self._interpolate(self.vc_x, self.vc_y)

    @staticmethod
    def _interpolate(x, y):
        rx, ry = [], []
        d_theta = 0.05
        for i in range(len(x) - 1):
            rx.extend([(1.0 - theta) * x[i] + theta * x[i + 1]
                       for theta in np.arange(0.0, 1.0, d_theta)])
            ry.extend([(1.0 - theta) * y[i] + theta * y[i + 1]
                       for theta in np.arange(0.0, 1.0, d_theta)])

        rx.extend([(1.0 - theta) * x[len(x) - 1] + theta * x[1]
                   for theta in np.arange(0.0, 1.0, d_theta)])
        ry.extend([(1.0 - theta) * y[len(y) - 1] + theta * y[1]
                   for theta in np.arange(0.0, 1.0, d_theta)])

        return rx, ry


class LidarSimulator:

    def __init__(self):
        self.range_noise = 0.01

    def get_observation_points(self, v_list: list[VehicleSimulator]) -> tuple[list[float], list[float], list[float], list[float], list[float]]:
        """
        Get observation points from a list of vehicles.

        Parameters
        ----------
        v_list : list[VehicleSimulator]
            List of vehicles

        Returns
        -------
        x : list[float]
            x positions of range points from an object
        y : list[float]
            y positions of range points from an object
        angle : list[float]
            angle of range points from an object
        r : list[float]
            range of range points from an object
        """
        x, y, angle, r = [], [], [], []

        # store all points
        for v in v_list:

            gx, gy = v.calc_global_contour()

            for vx, vy in zip(gx, gy):
                v_angle = math.atan2(vy, vx)
                vr = np.hypot(vx, vy) * random.uniform(1.0 - self.range_noise,
                                                       1.0 + self.range_noise)

                x.append(vx)
                y.append(vy)
                angle.append(v_angle)
                r.append(vr)

        return x, y, angle, r


class RectangleFitter:
    """
    LShapeFitting class. You can use this class by initializing the class and
    changing the parameters, and then calling the fitting method.
    """

    class Criteria(Enum):
        AREA = 1
        CLOSENESS = 2
        VARIANCE = 3

    def __init__(self):
        """
        Default parameter settings
        """
        #: Fitting criteria parameter
        self.criteria = self.Criteria.VARIANCE
        #: Minimum distance for closeness criteria parameter [m]
        self.min_dist_of_closeness_criteria = 0.01
        #: Angle difference parameter [deg]
        self.d_theta_deg_for_search = 1.0
        #: Range segmentation parameter [m]
        self.R0 = 3.0
        #: Range segmentation parameter [m]
        self.Rd = 0.001

    def fit(self, ox: list[float], oy: list[float]) -> tuple[list["RectangleData"], list[set[int]]]:
        """
        Fit L-shape rectangle model to object points.

        Parameters
        ----------
        ox : x positions of range points from an object
        oy : y positions of range points from an object

        Returns
        -------
        rects: Fitting rectangles
        id_sets: id sets of each cluster
        """
        # Perform range-based segmentation to a set of clusters,
        # where each cluster is an object to fit a rectangle to.
        id_sets = self._adoptive_range_segmentation(ox, oy)

        # For each cluster, find the best fitting rectangle.
        rects: list[RectangleData] = []
        for ids in id_sets:  # for each cluster
            cx, cy = zip(*[(ox[i], oy[i]) for i in ids])
            rects.append(self._rectangle_search(cx, cy))

        return rects, id_sets

    def _calc_area_criterion(self, c1, c2):
        c1_max, c1_min, c2_max, c2_min = RectangleFitter._find_min_max(c1, c2)
        alpha = -(c1_max - c1_min) * (c2_max - c2_min)
        return alpha

    def _calc_closeness_criterion(self, c1, c2):
        c1_max, c1_min, c2_max, c2_min = RectangleFitter._find_min_max(c1, c2)

        # Vectorization
        d1 = np.minimum(c1_max - c1, c1 - c1_min)
        d2 = np.minimum(c2_max - c2, c2 - c2_min)
        d = np.maximum(np.minimum(d1, d2), self.min_dist_of_closeness_criteria)
        beta = (1.0 / d).sum()

        return beta

    def _calc_variance_criterion(self, c1, c2):
        c1_max, c1_min, c2_max, c2_min = RectangleFitter._find_min_max(c1, c2)

        # Vectorization
        d1 = np.minimum(c1_max - c1, c1 - c1_min)
        d2 = np.minimum(c2_max - c2, c2 - c2_min)
        e1 = d1[d1 < d2]
        e2 = d2[d1 >= d2]
        v1 = - np.var(e1) if len(e1) > 0 else 0.0
        v2 = - np.var(e2) if len(e2) > 0 else 0.0
        gamma = v1 + v2

        return gamma

    @staticmethod
    def _find_min_max(c1, c2):
        c1_max = max(c1)
        c2_max = max(c2)
        c1_min = min(c1)
        c2_min = min(c2)
        return c1_max, c1_min, c2_max, c2_min

    def _adoptive_range_segmentation(self, ox: list[float], oy: list[float]) -> list[set[int]]:
        """
        Perform a range-based segmentation on a set of points in a 2D space.

        This method groups the points into clusters based on a dynamically calculated range.
        The range for a point is calculated as R0 + Rd * norm(point), where R0 and Rd are constants,
        and norm(point) is the Euclidean norm of the point.
        The method then merges any clusters that have any points in common.

        Args:
            ox (List[float]): The x-coordinates of the points.
            oy (List[float]): The y-coordinates of the points.

        Returns:
            List[Set[int]]: A list of sets, where each set represents a cluster of points.
            The integers in the sets are the indices of the points in the original lists.
        """
        # Segment list is a list of sets, where each set represents a cluster of points.
        segment_list: list[set[int]] = []

        # For each point...
        for point_x_1, point_y_1 in zip(ox, oy):
            cluster = set[int]()
            # Calculate the range for this point
            point_range: float = self.R0 + self.Rd * np.linalg.norm([point_x_1, point_y_1])

            # Calculate the distance between the current point and all other points
            for index, (point_x_2, point_y_2) in enumerate(zip(ox, oy)):
                distance: float = np.hypot(point_x_1 - point_x_2, point_y_1 - point_y_2)

                # If the distance is less than or equal to the range,
                # then add the point to the cluster
                if distance <= point_range:
                    cluster.add(index)

            segment_list.append(cluster)

        # Merge clusters
        while True:
            no_change: bool = True
            # Check all pairs of clusters
            for (cluster_index_1, cluster_index_2) in itertools.permutations(range(len(segment_list)), 2):
                # If the clusters have any points in common
                if segment_list[cluster_index_1] & segment_list[cluster_index_2]:
                    # Merge the clusters and remove the second one from the list
                    segment_list[cluster_index_1] = (segment_list[cluster_index_1] | segment_list.pop(cluster_index_2))
                    no_change = False
                    break
            # If no clusters were merged in this iteration, stop the loop
            if no_change:
                break

        return segment_list

    def _rectangle_search(self, x: list[float], y: list[float]) -> "RectangleData":
        """
        Find the best fitting rectangle for a given set of points in a 2D space.

        This function rotates the points and calculates a cost for each rotation based on a specified criterion.
        The rotation with the highest cost is selected as the best fit. The function then calculates the coordinates
        of the best fitting rectangle and returns it.

        Args:
            x (List[float]): The x-coordinates of the points.
            y (List[float]): The y-coordinates of the points.

        Returns:
            RectangleData: The best fitting rectangle for the given points.

        Notes:
            The rotation of points in the _rectangle_search function is a part
            of the process to find the best fitting rectangle for a given set
            of points in a 2D space.

            The idea is to evaluate how well a rectangle fits the points for
            various orientations of the rectangle. By rotating the points, the
            function effectively rotates the rectangle in the opposite direction.
            For each rotation, it calculates a cost based on a specified criterion
            (area, closeness, or variance). The rotation that gives the highest
            cost is selected as the best fit.

            This approach is based on the principle that a rectangle may fit
            the points better in some orientations than in others. For example,
            if the points form a long, narrow cluster, a rectangle that is aligned
            with the long axis of the cluster will fit the points better than a
            rectangle that is aligned with the short axis. By rotating the points
            (and thus the rectangle), the function can find the orientation that
            provides the best fit.
        """
        # Transpose the points to an array of (x, y) coordinates.
        xy = np.array([x, y]).T

        # The step size for the rotation.
        d_theta: float = np.deg2rad(self.d_theta_deg_for_search)

        # Initialize the minimum cost to negative infinity,
        # this tuple is (cost of the best fitting rectangle, angle of the best fitting rectangle).
        max_cost: tuple[float, float | None] = (-float('inf'), None)

        # Iterate over all possible rotations
        for theta in np.arange(0.0, np.pi / 2.0 - d_theta, d_theta):

            # Rotate the points
            c = xy @ rot_mat_2d(theta)  # Rotate the points by theta
            c1 = c[:, 0]  # All the x-coordinates
            c2 = c[:, 1]  # All the y-coordinates

            # Calculate the cost based on the specified criterion
            cost: float = 0.0
            if self.criteria == self.Criteria.AREA:
                cost = self._calc_area_criterion(c1, c2)
            elif self.criteria == self.Criteria.CLOSENESS:
                cost = self._calc_closeness_criterion(c1, c2)
            elif self.criteria == self.Criteria.VARIANCE:
                cost = self._calc_variance_criterion(c1, c2)

            # If the cost is greater than the current minimum cost, update the maximum cost.
            if cost > max_cost[0]:
                max_cost = (cost, theta)

        # Calculate the coordinates of the best fitting rectangle
        sin_s = np.sin(max_cost[1])
        cos_s = np.cos(max_cost[1])

        c1_s = xy @ np.array([cos_s, sin_s]).T
        c2_s = xy @ np.array([-sin_s, cos_s]).T

        a = [cos_s, -sin_s, cos_s, -sin_s]
        b = [sin_s, cos_s, sin_s, cos_s]
        c = [min(c1_s), min(c2_s), max(c1_s), max(c2_s)]
        return RectangleData(a=a, b=b, c=c)


class RectangleData:
    """A class to represent a rectangle defined by four lines."""

    def __init__(self, a: list[float], b: list[float], c: list[float]) -> None:
        """Initialize a new instance of the RectangleData class."""
        # Store the parameters of four lines (each represented by a, b, and c
        # in the line equation ax + by = c).
        self.a: list[float] = a
        self.b: list[float] = b
        self.c: list[float] = c

        # Stores the coordinates of the four corners of the rectangle,
        # plus a fifth point that is the same as the first point to
        # create a closed loop (mainly for plotting purposes).
        self.rect_coordinates_x: list[float] = [0.0] * 5
        self.rect_coordinates_y: list[float] = [0.0] * 5
        self.__calc_rect_contour()

    def plot(self):
        """Plot the rectangle."""
        plt.plot(self.rect_coordinates_x, self.rect_coordinates_y, "-r")

    def __calc_rect_contour(self):
        """Calculate the contour of the rectangle."""
        for i in range(4):
            self.rect_coordinates_x[i], self.rect_coordinates_y[i] = \
                self.__calc_cross_point(
                    [self.a[i], self.a[(i+1) % 4]],
                    [self.b[i], self.b[(i+1) % 4]],
                    [self.c[i], self.c[(i+1) % 4]]
                )
        self.rect_coordinates_x[4], self.rect_coordinates_y[4] = \
            self.rect_coordinates_x[0], self.rect_coordinates_y[0]

    @staticmethod
    def __calc_cross_point(a: list[float], b: list[float], c: list[float]) -> tuple[float, float]:
        """Calculate the intersection point of two lines.

        Args:
            a (List[float]): The 'a' coefficients of the lines.
            b (List[float]): The 'b' coefficients of the lines.
            c (List[float]): The 'c' coefficients of the lines.

        Returns:
            float, float: The x and y coordinates of the intersection point.
        """
        intersect_coordinates_x = (b[0] * -c[1] - b[1] * -c[0]) / (a[0] * b[1] - a[1] * b[0])
        intersect_coordinates_y = (a[1] * -c[0] - a[0] * -c[1]) / (a[0] * b[1] - a[1] * b[0])
        return intersect_coordinates_x, intersect_coordinates_y


# TODO: What if the points that are not hitting objects are not of infinite range value?
# How do we decide what a wall is? May need to add a max range threshold value instead of using inf.
# TODO: How do we get a 2D planar point cloud from a 3D point cloud? And how do we choose the height?
# TODO: May need to combine with circle fitting and grid map fitting to get this all to work and find
# both walls and objects in the environment
# TODO: If we combined with a vision based object detection and image segmentation algorithm, we could
# assign object IDs and segmented images with bounding boxes to the objects detected in the lidar data.
@staticmethod
def ray_casting_filter(theta_l: list[float], range_l: list[float], angle_resolution: float) -> tuple[list[float], list[float]]:
    """
    Filter a set of points using ray casting.

    Reduce the number of points in a 2D point cloud data. It does this by
    dividing the 2D space into sectors (like slices of a pie) and only
    keeping the point closest to the origin in each sector. This is often
    used in robotics, specifically in lidar data processing.

    Args:
        theta_l (list): The angles of the points from the origin.
        range_l (list): The distances of the points from the origin.
        angle_resolution (float): The size of each sector.

    Returns:
        list, list: The x and y coordinates of the filtered points.
    """
    inf = float("inf")
    rx: list[float] = []
    ry: list[float] = []

    # Initialize a list of infinite values, one for each sector.
    # The length of this list is determined by the angle_resolution parameter,
    # which specifies the size of each sector. The smaller the angle_resolution,
    # the more sectors there are.
    range_db = [inf for _ in range(
        int(np.floor((np.pi * 2.0) / angle_resolution)) + 1)]
    theta_db = [0.0 for _ in range(len(range_db))]

    # Iterate over all points.
    for i in range(len(theta_l)):  # pylint: disable=consider-using-enumerate
        # Calculate the id of the sector the point belongs to.
        angle_id = int(round(theta_l[i] / angle_resolution))

        print(f"angle_id: {angle_id}")
        print(f"Number of sectors: {len(range_db)}")

        # If the point is closer to the origin than the current closest point
        # in that sector, update the closest point.
        if range_db[angle_id] > range_l[i]:
            range_db[angle_id] = range_l[i]
            theta_db[angle_id] = theta_l[i]

    # Convert the filtered data back into Cartesian coordinates.
    for i in range(len(range_db)):
        t = theta_db[i]
        if range_db[i] != inf:
            # Use the polar to cartesian conversion formulas.
            rx.append(range_db[i] * np.cos(t))
            ry.append(range_db[i] * np.sin(t))

    return rx, ry


def main():

    # simulation parameters
    sim_time = 30.0  # simulation time
    dt = 0.2  # time tick

    angle_resolution = np.deg2rad(3.0)  # sensor angle resolution

    v1 = VehicleSimulator(-10.0, 0.0, np.deg2rad(90.0),
                          0.0, 50.0 / 3.6, 3.0, 5.0)
    v2 = VehicleSimulator(20.0, 10.0, np.deg2rad(180.0),
                          0.0, 50.0 / 3.6, 4.0, 10.0)

    l_shape_fitting = RectangleFitter()
    lidar_sim = LidarSimulator()

    time = 0.0
    while time <= sim_time:
        time += dt

        v1.update(dt, 0.1, 0.0)
        v2.update(dt, 0.1, -0.05)

        unfiltered_ox, unfiltered_oy, angles, ranges = lidar_sim.get_observation_points([v1, v2])
        ox, oy = ray_casting_filter(angles, ranges, angle_resolution)

        rects, id_sets = l_shape_fitting.fit(ox, oy)

        show_animation = True
        if show_animation:
            plt.cla()
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect(
                'key_release_event',
                lambda event: [exit(0) if event.key == 'escape' else None])
            plt.axis("equal")
            plt.plot(0.0, 0.0, "*r")
            v1.plot()
            v2.plot()

            for (ix, iy) in zip(unfiltered_ox, unfiltered_oy):
                plt.plot([0.0, ix], [0.0, iy], "-k")

            # Plot range observations.
            for ids in id_sets:
                x = [ox[i] for i in ids]
                y = [oy[i] for i in ids]

                # Plot the ray casts.
                for (ix, iy) in zip(x, y):
                    plt.plot([0.0, ix], [0.0, iy], "-og")

                # Plot the points.
                plt.plot(x, y, "o")
            for rect in rects:
                rect.plot()

            plt.pause(0.1)

    print("Done")


if __name__ == '__main__':
    main()
