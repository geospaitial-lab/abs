# @author: Maryniak, Marius - Fachbereich Elektrotechnik, Westfälische Hochschule Gelsenkirchen

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.interpolate import splprep, splev
from scipy.ndimage import uniform_filter1d
from scipy.spatial import KDTree

from src.GSC_Utils import LaserData


def get_df(cluster_id, sampling_points):
    """
    | Returns the dataframe of the sampling points of a cluster with the following schema:
    | cluster: id of the cluster
    | direction: id of the direction (0: longitudinal, 1: transverse)
    | lane: id of the lane (longitudinal: 0: left_2, 1: left_1, 2: mid, 3: right_1, 4: right_2, transverse: 0 to 24)
    | index: id of the index along the lanes
    | x: x coordinate of the point
    | y: y coordinate of the point
    | z: height in meters
    |
    | The shape of the longitudinal sampling points is (n, 5, 2) with n indices, 5 lanes and 2 coordinates.
    | The shape of the transverse sampling points is (n, 25, 2) with n indices, 25 lanes and 2 coordinates.

    :param int cluster_id: id of the cluster
    :param (np.ndarray[np.float64], np.ndarray[np.float64]) sampling_points: coordinates of the sampling points
        (longitudinal sampling points, transverse sampling points)
    :returns: dataframe
    :rtype: pd.DataFrame
    """
    num_sampling_points_longitudinal = sampling_points[0].shape[0]
    sampling_points_longitudinal = sampling_points[0].reshape((-1, 2))

    cluster_ids_longitudinal = np.full((num_sampling_points_longitudinal * 5), cluster_id)
    direction_ids_longitudinal = np.full((num_sampling_points_longitudinal * 5), 0)
    lane_ids_longitudinal = np.tile(np.arange(5), num_sampling_points_longitudinal)
    index_ids_longitudinal = np.repeat(np.arange(num_sampling_points_longitudinal), 5)
    placeholder_longitudinal = np.full((num_sampling_points_longitudinal * 5), np.nan)

    num_sampling_points_transverse = sampling_points[1].shape[0]
    sampling_points_transverse = sampling_points[1].reshape((-1, 2))

    cluster_ids_transverse = np.full((num_sampling_points_transverse * 25), cluster_id)
    direction_ids_transverse = np.full((num_sampling_points_transverse * 25), 1)
    lane_ids_transverse = np.tile(np.arange(25), num_sampling_points_transverse)
    index_ids_transverse = np.repeat(np.arange(num_sampling_points_transverse), 25)
    placeholder_transverse = np.full((num_sampling_points_transverse * 25), np.nan)

    recording_ids = np.concatenate((sampling_points[2].reshape(-1), sampling_points[3].reshape(-1)), axis=0)
    cluster_ids = np.concatenate((cluster_ids_longitudinal, cluster_ids_transverse), axis=0)
    direction_ids = np.concatenate((direction_ids_longitudinal, direction_ids_transverse), axis=0)
    lane_ids = np.concatenate((lane_ids_longitudinal, lane_ids_transverse), axis=0)
    index_ids = np.concatenate((index_ids_longitudinal, index_ids_transverse), axis=0)
    sampling_points = np.concatenate((sampling_points_longitudinal, sampling_points_transverse), axis=0)
    placeholder = np.concatenate((placeholder_longitudinal, placeholder_transverse), axis=0)

    sampling_points = {'recording_id': recording_ids,
                       'height_layer': placeholder,
                       'cluster': cluster_ids,
                       'direction': direction_ids,
                       'lane': lane_ids,
                       'index': index_ids,
                       'x': sampling_points[:, 0],
                       'y': sampling_points[:, 1],
                       'z': placeholder}

    return pd.DataFrame(sampling_points)


def get_gdf(clustered_sampling_points, crs='EPSG:25832'):
    """
    | Returns the geodataframe of the sampling points with the following schema:
    | cluster: id of the cluster
    | direction: id of the direction (0: longitudinal, 1: transverse)
    | lane: id of the lane (longitudinal: 0: left_2, 1: left_1, 2: mid, 3: right_1, 4: right_2, transverse: 0 to 24)
    | index: id of the index along the lanes
    | z: height in meters
    | geometry: x and y coordinates of the point
    |
    | The shape of the longitudinal sampling points of each cluster is (n, 5, 2) with n indices, 5 lanes and
        2 coordinates.
    | The shape of the transverse sampling points of each cluster is (n, 25, 2) with n indices, 25 lanes and
        2 coordinates.

    :param list[(np.ndarray[np.float64], np.ndarray[np.float64], np.ndarray[str] or None, np.ndarray[str] or None)]
        clustered_sampling_points: coordinates of the clustered sampling points (longitudinal sampling points,
        transverse sampling points and their corresponding panorama image ids)
    :param str crs: coordinate reference system
    :returns: geodataframe
    :rtype: gpd.GeoDataFrame
    """
    # noinspection PyTypeChecker
    clustered_dfs = [get_df(cluster_id=cluster_id,
                            sampling_points=sampling_points)
                     for cluster_id, sampling_points in enumerate(clustered_sampling_points)]

    df = pd.concat(clustered_dfs, ignore_index=True)

    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['x'], df['y']))
    gdf.drop(columns=['x', 'y'], inplace=True)
    gdf.set_crs(crs, inplace=True)

    return gdf


def get_recording_points(gdf,
                         id_column='image_id',
                         crs='EPSG:25832'):
    """
    | Returns the coordinates and the ids of each recording point.

    :param gpd.GeoDataFrame gdf: geodataframe
    :param str id_column: name of the id column ('image_id' for Cyclomedia data)
    :param str crs: coordinate reference system
    :returns: coordinates of the recording points and ids of the recording points
    :rtype: (np.ndarray[np.float64], np.ndarray[np.uint64])
    """
    gdf = gdf.to_crs(crs)

    recording_points = np.stack([gdf.geometry.x, gdf.geometry.y], axis=-1).astype(np.float64)
    recording_points_ids = np.array(gdf[id_column])

    return recording_points, recording_points_ids


def cluster_recording_points(recording_points,
                             recording_points_ids,
                             distance=5.,
                             distance_offset=.2,
                             base=36):
    """
    | Returns the coordinates of the clustered recording points.

    :param np.ndarray[np.float64] recording_points: coordinates of the recording points
    :param np.ndarray[np.uint64] recording_points_ids: ids of the recording points
    :param float distance: distance between consecutive recording points in meters (5.0 for Cyclomedia data)
    :param float distance_offset: distance offset between consecutive recording points in percent
    :param int base: base of the id (36 for Cyclomedia data)
    :returns: list of coordinates of the clustered recording points, list of ids of the clustered recording points
    :rtype: (list[np.ndarray[np.float64]], list[np.ndarray[str]])
    """
    sorted_indices = np.argsort(recording_points_ids)
    recording_points = recording_points[sorted_indices]
    recording_points_ids = recording_points_ids[sorted_indices]

    if base != 10:
        base_to_base10 = np.vectorize(lambda x: int(x, base))
        recording_points_ids_base10 = base_to_base10(recording_points_ids).astype(np.uint64)
    else:
        recording_points_ids_base10 = recording_points_ids.astype(np.uint64)

    split_indices_id = np.where(np.diff(recording_points_ids_base10) > 1)[0] + 1

    distances = np.linalg.norm(np.diff(recording_points, axis=0), axis=-1)
    distance_min = distance * (1 - distance_offset)
    distance_max = distance * (1 + distance_offset)
    split_indices_distance = np.where(np.logical_or(distances < distance_min, distances > distance_max))[0] + 1

    split_indices = np.union1d(split_indices_id, split_indices_distance)
    clustered_recording_points = np.split(recording_points, split_indices)
    clustered_recording_points_ids = np.split(recording_points_ids, split_indices)

    clustered_recording_points_ids = [recording_points_ids
                                      for recording_points, recording_points_ids
                                      in zip(clustered_recording_points, clustered_recording_points_ids)
                                      if recording_points.shape[0] > 1]

    clustered_recording_points = [recording_points
                                  for recording_points in clustered_recording_points
                                  if recording_points.shape[0] > 1]

    return clustered_recording_points, clustered_recording_points_ids


def interpolate_pseudo_recording_points(line, distance):
    """
    | Returns the coordinates of the interpolated pseudo recording points.

    :param LineString line: line
    :param int distance: distance between consecutive pseudo recording points in meters (5 for Cyclomedia data)
    :returns: coordinates of the interpolated pseudo recording points
    :rtype: np.ndarray[np.float64]
    """
    pseudo_recording_points = [(line.interpolate(distance).x, line.interpolate(distance).y)
                               for distance in range(0, int(line.length), distance)]
    return np.array(pseudo_recording_points, dtype=np.float64)


def cluster_pseudo_recording_points(gdf_pseudo_lanes, distance=5):
    """
    | Returns the coordinates of the clustered pseudo recording points.

    :param gpd.GeoDataFrame gdf_pseudo_lanes: geodataframe of pseudo lanes
    :param int distance: distance between consecutive recording points in meters (5 for Cyclomedia data)
    :returns: list of coordinates of the clustered recording points, list of ids of the clustered recording points
    :rtype: list[np.ndarray[np.float64]]
    """
    clustered_pseudo_recording_points = [interpolate_pseudo_recording_points(line, distance)
                                         for line in gdf_pseudo_lanes.geometry]

    return clustered_pseudo_recording_points


def get_sampling_points(recording_points,
                        recording_points_ids,
                        lane_offset=.75,
                        extrapolation_distance=1.5):
    """
    | Longitudinal direction:
    | Returns the coordinates of the sampling points with a sampling distance of 10cm along the lane
        of the right wheels. The lane is offset to the right from the lane specified by the recording points.
    | Four additional lanes will be added, with two lanes on the left and two lanes on the right of the main lane.
        The distance between each lane is 10cm.
    | Each lane is tangentially extrapolated at the start and at the end.
    | Transverse direction:
    | Returns the coordinates of the sampling points with a sampling distance of 50cm along the lane
        specified by the recording points.
    | 24 additional lanes will be added, with 12 lanes on the left and 12 lanes on the right of the main lane.
        The distance between each lane is 10cm.
    | Each lane is tangentially extrapolated at the start and at the end.

    :param np.ndarray[np.float64] recording_points: coordinates of the recording points
    :param np.ndarray[str] or None recording_points_ids: ids of the recording points
    :param float lane_offset: lane offset between the lane specified by the recording points and the lane
        of the right wheels in meters
    :param float extrapolation_distance: distance of the extrapolation at the start and at the end of each lane
        in meters
    :returns: coordinates of the sampling points of each lane (longitudinal) and coordinates of the sampling points of
        each lane (transverse) and their corresponding panorama image ids (ids is None if they are pseudo
        recording points)
    :rtype: (np.ndarray[np.float64], np.ndarray[np.float64], np.ndarray[str], np.ndarray[str])
    """
    if recording_points.shape[0] < 4:
        k = 1  # linear spline interpolation
    else:
        k = 3  # cubic spline interpolation

    x = recording_points[:, 0]
    y = recording_points[:, 1]
    tck, u, *_ = splprep([x, y], k=k, s=0)

    if recording_points_ids is not None:
        u_tree = KDTree(u[:, np.newaxis])

    t = np.linspace(0, 1, 1000)
    curve_points = np.array(splev(t, tck, ext=2)).T
    curve_length = np.sum(np.linalg.norm(np.diff(curve_points, axis=0), axis=1))

    num_sampling_points = int(curve_length / .1) + 1
    t = np.linspace(0, 1, num_sampling_points)
    sampling_points_mid = np.array(splev(t, tck, ext=2)).T

    tangent_vectors = np.array(splev(t, tck, der=1)).T
    tangent_vectors /= np.linalg.norm(tangent_vectors, axis=1, keepdims=True)
    normal_vectors = np.array([-tangent_vectors[:, 1], tangent_vectors[:, 0]]).T

    lane_offset_factors = np.array([-.2, -.1, 0., .1, .2]) + lane_offset
    sampling_points_longitudinal = (sampling_points_mid[:, np.newaxis, :] -
                                    lane_offset_factors[np.newaxis, :, np.newaxis] * normal_vectors[:, np.newaxis])

    extrapolated_sampling_points_longitudinal_start = \
        (sampling_points_longitudinal[0][np.newaxis, ...] -
         (tangent_vectors[0] *
          np.arange(.1, extrapolation_distance + .1, .1)[..., np.newaxis])[:, np.newaxis, :])
    extrapolated_sampling_points_longitudinal_start = np.flip(extrapolated_sampling_points_longitudinal_start, axis=0)

    extrapolated_sampling_points_longitudinal_end = \
        (sampling_points_longitudinal[-1][np.newaxis, ...] +
         (tangent_vectors[-1] *
          np.arange(.1, extrapolation_distance + .1, .1)[..., np.newaxis])[:, np.newaxis, :])

    sampling_points_longitudinal = np.concatenate((extrapolated_sampling_points_longitudinal_start,
                                                   sampling_points_longitudinal,
                                                   extrapolated_sampling_points_longitudinal_end),
                                                  axis=0)

    if recording_points_ids is not None:
        t_longitudinal_extrapolated = \
            np.concatenate((np.zeros(extrapolated_sampling_points_longitudinal_start.shape[0]),
                            t,
                            np.ones(extrapolated_sampling_points_longitudinal_end.shape[0])))

        _, indices_longitudinal = u_tree.query(t_longitudinal_extrapolated[:, np.newaxis],
                                               k=1,
                                               workers=-1)

        ids_longitudinal = recording_points_ids[indices_longitudinal]
        ids_longitudinal = np.repeat(ids_longitudinal, lane_offset_factors.shape[0])
        ids_longitudinal = ids_longitudinal.reshape(-1, lane_offset_factors.shape[0])
    else:
        ids_longitudinal = np.empty(sampling_points_longitudinal.shape[:-1], dtype=np.dtype('U7'))
        ids_longitudinal.fill('_pseudo')

    lane_offset_factors = np.arange(-1.2, 1.2 + .1, .1)
    sampling_points_transverse = (sampling_points_mid[:, np.newaxis, :] -
                                  lane_offset_factors[np.newaxis, :, np.newaxis] * normal_vectors[:, np.newaxis])

    extrapolated_sampling_points_transverse_start = \
        (sampling_points_transverse[0][np.newaxis, ...] -
         (tangent_vectors[0] *
          np.arange(.1, extrapolation_distance + .1, .1)[..., np.newaxis])[:, np.newaxis, :])
    extrapolated_sampling_points_transverse_start = np.flip(extrapolated_sampling_points_transverse_start, axis=0)

    extrapolated_sampling_points_transverse_end = \
        (sampling_points_transverse[-1][np.newaxis, ...] +
         (tangent_vectors[-1] *
          np.arange(.1, extrapolation_distance + .1, .1)[..., np.newaxis])[:, np.newaxis, :])

    sampling_points_transverse = np.concatenate((extrapolated_sampling_points_transverse_start,
                                                 sampling_points_transverse,
                                                 extrapolated_sampling_points_transverse_end),
                                                axis=0)
    sampling_points_transverse = sampling_points_transverse[::5]

    if recording_points_ids is not None:
        t_transverse_extrapolated = \
            np.concatenate((np.zeros(extrapolated_sampling_points_transverse_start.shape[0]),
                            t,
                            np.ones(extrapolated_sampling_points_transverse_end.shape[0])))
        t_transverse_extrapolated = t_transverse_extrapolated[::5]

        _, indices_transverse = u_tree.query(t_transverse_extrapolated[:, np.newaxis],
                                             k=1,
                                             workers=-1)

        ids_transverse = recording_points_ids[indices_transverse]
        ids_transverse = np.repeat(ids_transverse, lane_offset_factors.shape[0])
        ids_transverse = ids_transverse.reshape(-1, lane_offset_factors.shape[0])
    else:
        ids_transverse = np.empty(sampling_points_transverse.shape[:-1], dtype=np.dtype('U7'))
        ids_transverse.fill('_pseudo')

    return sampling_points_longitudinal, sampling_points_transverse, ids_longitudinal, ids_transverse


def get_clustered_sampling_points(clustered_recording_points,
                                  clustered_recording_points_ids,
                                  lane_offset=.75,
                                  extrapolation_distance=1.5):
    """
    | Applies get_sampling_points() to each cluster of recording points.
    | Returns the coordinates of the clustered sampling points.

    :param list[np.ndarray[np.float64]] clustered_recording_points: coordinates of the clustered recording points
    :param list[np.ndarray[str]] or None clustered_recording_points_ids: list of ids of the clustered recording points
    :param float lane_offset: lane offset between the lane specified by the recording points and the lane
        of the right wheels in meters
    :param float extrapolation_distance: distance of the extrapolation at the start and at the end of each lane
        in meters
    :returns: list of coordinates of the clustered sampling points and their corresponding panorma image ids
    :rtype: list[(np.ndarray[np.float64], np.ndarray[np.float64], np.ndarray[str], np.ndarray[str])]
    """
    if clustered_recording_points_ids is None:
        clustered_recording_points_ids = [None] * len(clustered_recording_points)

    clustered_sampling_points = [get_sampling_points(recording_points=recording_points,
                                                     recording_points_ids=recording_points_ids,
                                                     lane_offset=lane_offset,
                                                     extrapolation_distance=extrapolation_distance)
                                 for recording_points, recording_points_ids
                                 in zip(clustered_recording_points, clustered_recording_points_ids)]

    return clustered_sampling_points


def create_planeness_sampling_points(gdf,
                                     id_column,
                                     id_base=36,
                                     gdf_pseudo_lanes=None,
                                     crs='EPSG:25832'):
    """
    | Returns the coordinates of the planeness sampling points.

    :param gpd.GeoDataFrame gdf: geodataframe of the recording points
    :param str id_column: name of the id column ('image_id' for Cyclomedia data)
    :param int id_base: base of the id (36 for Cyclomedia data)
    :param gpd.GeoDataFrame or None gdf_pseudo_lanes: geodataframe of pseudo lanes
    :param str crs: coordinate reference system
    :returns: geodataframe of the planeness sampling points
    :rtype: gpd.GeoDataFrame
    """
    recording_points, recording_points_ids = get_recording_points(gdf=gdf,
                                                                  id_column=id_column,
                                                                  crs=crs)

    clustered_recording_points, clustered_recording_points_ids = \
        cluster_recording_points(recording_points=recording_points,
                                 recording_points_ids=recording_points_ids,
                                 distance=5.,
                                 distance_offset=.2,
                                 base=id_base)

    clustered_sampling_points = \
        get_clustered_sampling_points(clustered_recording_points=clustered_recording_points,
                                      clustered_recording_points_ids=clustered_recording_points_ids,
                                      lane_offset=.75,
                                      extrapolation_distance=1.5)

    if gdf_pseudo_lanes is not None:
        clustered_pseudo_recording_points = cluster_pseudo_recording_points(gdf_pseudo_lanes, distance=5)

        clustered_pseudo_sampling_points = \
            get_clustered_sampling_points(clustered_recording_points=clustered_pseudo_recording_points,
                                          clustered_recording_points_ids=None,
                                          lane_offset=.75,
                                          extrapolation_distance=1.5)

        clustered_sampling_points += clustered_pseudo_sampling_points

    planeness_sampling_points = get_gdf(clustered_sampling_points=clustered_sampling_points,
                                        crs=crs)

    return planeness_sampling_points


def sample_points(laserdata,
                  planeness_sampling_points,
                  queried_neighbors=10,
                  radius=.025,
                  power=2):
    """
    | Returns the heights of the sampling points.
    | Inverse distance weighting is used for interpolation.

    :param LaserData laserdata: laserdata
    :param gpd.GeoDataFrame planeness_sampling_points: coordinates of the sampling points
    :param int queried_neighbors: number of queried neighbors
    :param float radius: sampling radius in meters
    :param int power: power parameter
    :returns: heights of the sampling points
    :rtype: np.ndarray[np.float64]
    """
    point_cloud = np.stack((laserdata.x, laserdata.y), axis=-1)
    k_d_tree = KDTree(point_cloud)

    sampling_points = np.concatenate((np.array(planeness_sampling_points.geometry.x)[..., np.newaxis],
                                      np.array(planeness_sampling_points.geometry.y)[..., np.newaxis]),
                                     axis=1)

    distances, indices = k_d_tree.query(sampling_points,
                                        k=queried_neighbors,
                                        distance_upper_bound=radius,  # set distances larger than radius to np.inf and
                                        # their indices to the number of points of the point cloud
                                        workers=-1)

    indices[indices == len(laserdata)] = -1
    heights = laserdata.z[indices]

    with np.errstate(divide='ignore', invalid='ignore'):  # if distance or the sum of weights is 0
        weights = np.power(distances, -power)
        z_values = np.sum(weights * heights, axis=-1) / np.sum(weights, axis=-1)

    return z_values


def simulate_straight_edge(heights,
                           distances,
                           index_mid,
                           window_size):
    """
    | Returns the distances between the straight edge and the heights of the sampling points, the gradients
        and the indices of the max distances on the left and the right side of the straight edge.
    | The shape of the distances is (m, window_size, n) with m values (windows), window_size sampling points
        and n lanes.
    | The shape of the gradients is (m, n) with m values (windows) and n lanes.
    | The shape of the indices of the max distances is (m, n, 2) with m values (windows), n lanes and 2 indices.
    |
    | The shape of the heights and distances is (m, window_size, n) with m values (windows), window_size sampling points
        and n lanes.

    :param np.ndarray[np.float64] heights: heights
    :param np.ndarray[np.float64] distances: distances of the previous iteration or heights (first iteration)
    :param int index_mid: index of the middle of a window
    :param int window_size: odd window size (length of the straight edge in sampling points)
    :returns: distances, gradients and indices of the max distances
    :rtype: (np.ndarray[np.float64], np.ndarray[np.float64], np.ndarray[np.float64])
    """
    distances_left_max_indices = np.argmax(distances[:, :index_mid + 1], axis=-2)
    distances_right_max_indices = index_mid - np.argmax(np.flip(distances[:, index_mid:], axis=-2), axis=-2)  # last
    # occurrence

    distances_max_indices = np.concatenate((distances_left_max_indices[..., np.newaxis],
                                            distances_right_max_indices[..., np.newaxis] + index_mid),
                                           axis=-1)

    distances_left_max = np.take_along_axis(heights[:, :index_mid + 1],
                                            distances_left_max_indices[:, np.newaxis],
                                            axis=-2)
    distances_right_max = np.take_along_axis(heights[:, index_mid:],
                                             distances_right_max_indices[:, np.newaxis],
                                             axis=-2)

    with np.errstate(divide='ignore', invalid='ignore'):  # if the shared height in the middle of a window
        # is the max height
        gradients = ((distances_right_max - distances_left_max) /
                     ((distances_right_max_indices[:, np.newaxis] + index_mid) -
                      distances_left_max_indices[:, np.newaxis]))

    gradients = np.where(np.logical_or(np.isnan(gradients), np.isinf(gradients)),
                         0,
                         gradients)  # if the shared height in the middle of a window is the max height
    y_intercepts = distances_left_max - gradients * distances_left_max_indices[:, np.newaxis]

    ys = gradients * np.arange(window_size)[np.newaxis, :, np.newaxis] + y_intercepts
    distances = heights - ys

    return distances, gradients[:, 0, :] * 100, distances_max_indices


def stop_condition(distances):
    """
    | Stop condition (all distances are equal to 0 or less than 0) for simulate_straight_edge().
    |
    | The shape of the distances is (m, window_size, n) with m values (windows), window_size sampling points
        and n lanes.

    :param np.ndarray[np.float64] distances: distances of the previous iteration or heights (first iteration)
    :returns: True, if all distances are equal to 0 or less than 0
    :rtype: bool
    """
    return np.amax(distances) < 1e-12


def get_planeness_values_longitudinal(heights, window_size=41):
    """
    | Returns the longitudinal planeness values by simulating a rolling straight edge in longitudinal direction
        along all lanes.
    | The shape of the longitudinal planeness values is (n, 6) with n sampling points and 6 planeness values
        (depth_mid, delta_depth_mid_moving_average_11, delta_depth_mid_moving_average_31,
        delta_depth_mid_moving_average_101, delta_depth_mid_moving_average_301, gradient).
    |
    | The shape of the heights is (n, 5) with n indices and 5 lanes.
    |
    | Based on:
    | https://stackoverflow.com/a/43200476

    :param np.ndarray[np.float64] heights: heights
    :param int window_size: odd window size (length of the rolling straight edge in sampling points)
    :returns: longitudinal planeness values
    :rtype: np.ndarray[np.float64]
    """
    if np.all(np.isnan(heights)):
        heights = np.ones_like(heights)

    window_indices = (np.arange(window_size)[np.newaxis, ...] +
                      (np.arange(heights.shape[0] - window_size + 1))[np.newaxis, ...].T)
    heights = heights[window_indices]

    index_mid = window_size // 2

    distances = np.copy(heights)

    n = 0

    while not stop_condition(distances):
        distances, gradients, _ = simulate_straight_edge(heights=heights,
                                                         distances=distances,
                                                         index_mid=index_mid,
                                                         window_size=window_size)

        n += 1

        if n > 5:
            non_zero_indices = np.where(np.any(distances >= 1e-12, axis=(1, 2)))[0]

            for non_zero_index in non_zero_indices:
                distances_row = distances[non_zero_index][np.newaxis, ...]

                heights_row = np.copy(distances_row)

                while not stop_condition(distances_row):
                    distances_row, gradients_row, _ = simulate_straight_edge(heights=heights_row,
                                                                             distances=distances_row,
                                                                             index_mid=index_mid,
                                                                             window_size=window_size)

                distances[non_zero_index] = distances_row.squeeze()
                # noinspection PyUnboundLocalVariable
                gradients[non_zero_index] = gradients_row.squeeze()

    depths_mid = np.absolute(distances[:, index_mid, :])

    depths_mid_moving_average_11 = uniform_filter1d(depths_mid,
                                                    size=11,
                                                    axis=0,
                                                    mode='constant')
    depths_mid_moving_average_11[:5, :] = depths_mid_moving_average_11[-5:, :] = np.nan
    deltas_depths_mid_moving_average_11 = depths_mid - depths_mid_moving_average_11

    depths_mid_moving_average_31 = uniform_filter1d(depths_mid,
                                                    size=31,
                                                    axis=0,
                                                    mode='constant')
    depths_mid_moving_average_31[:15, :] = depths_mid_moving_average_31[-15:, :] = np.nan
    deltas_depths_mid_moving_average_31 = depths_mid - depths_mid_moving_average_31

    if depths_mid.shape[0] > 100:
        depths_mid_moving_average_101 = uniform_filter1d(depths_mid,
                                                         size=101,
                                                         axis=0,
                                                         mode='constant')
        depths_mid_moving_average_101[:50, :] = depths_mid_moving_average_101[-50:, :] = np.nan
        deltas_depths_mid_moving_average_101 = depths_mid - depths_mid_moving_average_101
    else:
        deltas_depths_mid_moving_average_101 = np.empty(depths_mid.shape)
        deltas_depths_mid_moving_average_101.fill(np.nan)

    if depths_mid.shape[0] > 300:
        depths_mid_moving_average_301 = uniform_filter1d(depths_mid,
                                                         size=301,
                                                         axis=0,
                                                         mode='constant')
        depths_mid_moving_average_301[:150, :] = depths_mid_moving_average_301[-150:, :] = np.nan
        deltas_depths_mid_moving_average_301 = depths_mid - depths_mid_moving_average_301
    else:
        deltas_depths_mid_moving_average_301 = np.empty(depths_mid.shape)
        deltas_depths_mid_moving_average_301.fill(np.nan)

    buffer = np.full((((window_size - 1) // 2), 5), np.nan)

    depths_mid = np.concatenate((buffer,
                                 depths_mid,
                                 buffer),
                                axis=0).reshape(-1)

    deltas_depths_mid_moving_average_11 = np.concatenate((buffer,
                                                          deltas_depths_mid_moving_average_11,
                                                          buffer),
                                                         axis=0).reshape(-1)

    deltas_depths_mid_moving_average_31 = np.concatenate((buffer,
                                                          deltas_depths_mid_moving_average_31,
                                                          buffer),
                                                         axis=0).reshape(-1)

    deltas_depths_mid_moving_average_101 = np.concatenate((buffer,
                                                           deltas_depths_mid_moving_average_101,
                                                           buffer),
                                                          axis=0).reshape(-1)

    deltas_depths_mid_moving_average_301 = np.concatenate((buffer,
                                                           deltas_depths_mid_moving_average_301,
                                                           buffer),
                                                          axis=0).reshape(-1)

    # noinspection PyUnboundLocalVariable
    gradients = np.concatenate((buffer,
                                gradients,
                                buffer),
                               axis=0).reshape(-1)

    planeness_values = np.concatenate((depths_mid[:, np.newaxis],
                                       deltas_depths_mid_moving_average_11[:, np.newaxis],
                                       deltas_depths_mid_moving_average_31[:, np.newaxis],
                                       deltas_depths_mid_moving_average_101[:, np.newaxis],
                                       deltas_depths_mid_moving_average_301[:, np.newaxis],
                                       gradients[:, np.newaxis]),
                                      axis=1)

    return planeness_values


def get_clustered_planeness_values_longitudinal(clustered_heights, window_size=41):
    """
    | Applies get_planeness_values_longitudinal() to each cluster of heights.
    | Returns the longitudinal planeness values.
    | The shape of the longitudinal planeness values is (n, 6) with n sampling points and 6 planeness values
        (depth_mid, delta_depth_mid_moving_average_11, delta_depth_mid_moving_average_31,
        delta_depth_mid_moving_average_101, delta_depth_mid_moving_average_301, gradient).
    |
    | The shape of the heights of each cluster is (n, 5) with n indices and 5 lanes.

    :param list[np.ndarray[np.float64]] clustered_heights: clustered heights
    :param int window_size: odd window size (length of the rolling straight edge in sampling points)
    :returns: longitudinal planeness values
    :rtype: np.ndarray[np.float64]
    """
    clustered_planeness_values = [get_planeness_values_longitudinal(heights=heights,
                                                                    window_size=window_size)
                                  for heights in clustered_heights]

    planeness_values = np.concatenate(clustered_planeness_values, axis=0)

    return planeness_values


def get_gradients_transverse(heights):
    """
    | Returns the gradients of the heights by using linear regression (note that the sign is flipped).
    | The shape of the gradients is (n) with n gradients.
    |
    | The shape of the heights is (n, 25) with n indices and 25 lanes.

    :param np.ndarray[np.float64] heights: heights
    :returns: gradients
    :rtype: np.ndarray[np.float64]
    """
    heights_indices = np.arange(25)
    heights_indices_mean = np.mean(heights_indices)

    heights_mean = np.mean(heights, axis=-1)

    gradients = (np.sum((heights_indices - heights_indices_mean) * (heights - heights_mean[:, np.newaxis]), axis=-1) /
                 np.sum((heights_indices - heights_indices_mean) ** 2))

    return - gradients * 100


def get_rut_depths_water_depths(heights,
                                distances,
                                distances_max_indices,
                                index):
    """
    | Returns the rut depths and the water depths of the heights for the left and the right lane.
    | The shape of the rut depths and water depths is (4) with 4 planeness values (rut_depth_left, rut_depth_right,
        water_depth_left, water_depth_right).
    |
    | The shape of the heights and distances is (n, 13, 2) with n indices, 13 lanes and 2 lanes (left and right).
    | The shape of the indices of the max distances is (n, 2, 2) with n indices, 2 lanes and 2 indices.

    :param np.ndarray[np.float64] heights: heights
    :param np.ndarray[np.float64] distances: distances
    :param np.ndarray[np.float64] distances_max_indices: indices of the max distances
    :param int index: index
    :returns: rut depths and water depths
    :rtype: np.ndarray[np.float64]
    """
    distances_left_lane = distances[index,
                          distances_max_indices[index, 0, 0]:distances_max_indices[index, 0, 1] + 1,
                          0]
    distances_right_lane = distances[index,
                           distances_max_indices[index, 1, 0]:distances_max_indices[index, 1, 1] + 1,
                           1]

    rut_depth_left = np.absolute(np.amin(distances_left_lane, axis=0))
    rut_depth_right = np.absolute(np.amin(distances_right_lane, axis=0))

    heights_left_lane = heights[index,
                        distances_max_indices[index, 0, 0]:distances_max_indices[index, 0, 1] + 1,
                        0]
    heights_right_lane = heights[index,
                         distances_max_indices[index, 1, 0]:distances_max_indices[index, 1, 1] + 1,
                         1]

    heights_left_lane_left_max = np.amax(heights_left_lane[:np.argmin(heights_left_lane, axis=0) + 1], axis=0)
    heights_left_lane_right_max = np.amax(heights_left_lane[np.argmin(heights_left_lane, axis=0):], axis=0)
    heights_right_lane_left_max = np.amax(heights_right_lane[:np.argmin(heights_right_lane, axis=0) + 1], axis=0)
    heights_right_lane_right_max = np.amax(heights_right_lane[np.argmin(heights_right_lane, axis=0):], axis=0)

    water_depth_left = np.absolute((np.amin((heights_left_lane_left_max, heights_left_lane_right_max)) -
                                    np.amin(heights_left_lane, axis=0)))
    water_depth_right = np.absolute((np.amin((heights_right_lane_left_max, heights_right_lane_right_max)) -
                                     np.amin(heights_right_lane, axis=0)))

    return np.array([rut_depth_left, rut_depth_right, water_depth_left, water_depth_right])


def get_planeness_values_transverse(heights):
    """
    | Returns the transverse planeness values by simulating a straight edge in transverse direction
        for the left and the right lane.
    | The shape of the transverse planeness values is (n, 5) with n sampling lines and 5 planeness values
        (rut_depth_left, rut_depth_right, water_depth_left, water_depth_right, gradient).
    |
    | The shape of the heights is (n, 25) with n indices and 25 lanes.

    :param np.ndarray[np.float64] heights: heights
    :returns: transverse planeness values
    :rtype: np.ndarray[np.float64]
    """
    if np.all(np.isnan(heights)):
        heights = np.ones_like(heights)

    gradients = get_gradients_transverse(heights=heights)

    heights = np.concatenate((heights[:, :13, np.newaxis],
                              heights[:, 12:, np.newaxis]),
                             axis=-1)

    distances = np.copy(heights)

    n = 0

    while not stop_condition(distances):
        distances, _, distances_max_indices = simulate_straight_edge(heights=heights,
                                                                     distances=distances,
                                                                     index_mid=6,
                                                                     window_size=13)

        n += 1

        if n > 5:
            non_zero_indices = np.where(np.any(distances >= 1e-12, axis=(1, 2)))[0]

            for non_zero_index in non_zero_indices:
                distances_row = distances[non_zero_index][np.newaxis, ...]

                heights_row = np.copy(distances_row)

                while not stop_condition(distances_row):
                    distances_row, _, distances_max_indices_row = simulate_straight_edge(heights=heights_row,
                                                                                         distances=distances_row,
                                                                                         index_mid=6,
                                                                                         window_size=13)

                distances[non_zero_index] = distances_row.squeeze()
                # noinspection PyUnboundLocalVariable
                distances_max_indices[non_zero_index] = distances_max_indices_row.squeeze()

    # noinspection PyUnboundLocalVariable
    rut_depths_water_depths = [get_rut_depths_water_depths(heights=heights,
                                                           distances=distances,
                                                           distances_max_indices=distances_max_indices,
                                                           index=index)
                               for index, _ in enumerate(distances)]

    planeness_values = np.concatenate((np.array(rut_depths_water_depths),
                                       gradients[:, np.newaxis]),
                                      axis=1)

    return planeness_values


def get_clustered_planeness_values_transverse(clustered_heights):
    """
    | Applies get_planeness_values_transverse() to each cluster of heights.
    | Returns the transverse planeness values.
    | The shape of the transverse planeness values is (n, 5) with n sampling lines and 5 planeness values
        (rut_depth_left, rut_depth_right, water_depth_left, water_depth_right, gradient).
    |
    | The shape of the heights of each cluster is (n, 25) with n indices and 25 lanes.

    :param list[np.ndarray[np.float64]] clustered_heights: clustered heights
    :returns: transverse planeness values
    :rtype: np.ndarray[np.float64]
    """
    clustered_planeness_values = [get_planeness_values_transverse(heights=heights)
                                  for heights in clustered_heights]

    planeness_values = np.concatenate(clustered_planeness_values, axis=0)

    return planeness_values
