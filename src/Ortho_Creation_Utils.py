# @author: Roß, Alexander - Fachbereich Elektrotechnik, Westfälische Hochschule Gelsenkirchen
# @coauthor: Maryniak, Marius - Fachbereich Elektrotechnik, Westfälische Hochschule Gelsenkirchen

import imghdr
import math
import multiprocessing
import os
import warnings

import cv2
import numpy as np
import pyproj
from astropy.time import Time
from dateutil.parser import isoparse
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from turbojpeg import TurboJPEG

from src.GSC_Utils import LaserData

tile_buffer = 7.5


def classify_outliers_in_laserdata(points, radius=0.5, threshold=100, subsample_by=5):
    """
    Classify outliers. All Points with less than a threshold of neighbouring points within a set radius are considered
    outliers.

    :param LaserData points: LaserData object with the points to classify.
    :param float radius: Radius to consider neighbouring points within.
    :param int threshold: Minimal number of neighbouring points within radius for inlier.
    :param int subsample_by: Factor to subsample pointcloud to count neighbours in by. Speeds up computations.
    :return: A ndarray of shape (len(points), ) containing True at the position of outliers and False otherwise.
    :rtype: np.ndarray of bool
    """
    if subsample_by == 0:
        raise ValueError("0 is not a valid subsample factor!")

    if threshold == 0:
        return np.zeros(len(points)).astype(bool)

    if subsample_by > threshold:
        warnings.warn(f"Subsampling factor too big for threshold! Using new factor {threshold}!")
        subsample_by = threshold

    adjusted_threshold = threshold // subsample_by

    points = np.stack((points.x, points.y, points.z), axis=-1)

    subsampled_points = points[::subsample_by]

    tree = cKDTree(subsampled_points)
    dists, _ = tree.query(points, k=adjusted_threshold, workers=-1)
    if adjusted_threshold == 1:
        dists = dists[..., np.newaxis]

    classifications = np.any(dists >= radius, axis=-1)

    return classifications


def process_subtile(bin_heights, bin_idx, bin_size=.02, bin_range=5,
                    first_bin_threshold=10, n_bins_after_first_bin=10):
    bin_edges = np.arange(np.min(bin_heights), np.max(bin_heights) + bin_size, bin_size)
    if len(bin_edges) < 2:
        return bin_idx
    else:
        hist, _ = np.histogram(bin_heights, bins=bin_edges)

        first_bin_idx = np.argmax(hist > first_bin_threshold)
        max_bin_idx = np.argmax(hist[first_bin_idx:first_bin_idx + n_bins_after_first_bin]) + first_bin_idx

        max_bin_start = bin_edges[np.clip(max_bin_idx - (bin_range // 2), 0, hist.shape[0])]
        max_bin_end = bin_edges[np.clip(max_bin_idx + (bin_range // 2) + 1, 0, hist.shape[0])]

        points_idx_in_max_bin = bin_idx[(bin_heights >= max_bin_start) & (bin_heights < max_bin_end)]

        return points_idx_in_max_bin


def classify_ground(points, bounds, bin_per_dim, bin_size=.02, bin_range=5,
                    first_bin_threshold=10, n_bins_after_first_bin=10):
    """
    Classify points into ground and non-ground points. Points are classified with a simple approach, that works good
    enough for the intended purpose. First the points are gridded in the xy-plane. Then for each grid-cell all points
    above a given threshold above the lowest point are classified as non-ground.

    :param LaserData points: LaserData object with the points to classify.
    :param tuple or list or np.ndarray bounds: Structure containing the bounds of the laz-file in the xy-plane.
    :param int bin_per_dim: Int specifying the number of grid-cells per direction.
    :param float bin_size: Bin size of per subtile height histogram
    :param int bin_range: Number of bins around the maximum bin in per subtile height histogram
        to consider as ground (odd)
    :param int first_bin_threshold: First bin to surpass this threshold is the start of the range to find maximum bin in
    :param int n_bins_after_first_bin: Length of the range to find the maximum bin in

    :return: A ndarray of shape (len(points), ) containing 1 at the position of ground-points and 0 otherwise.
    :rtype: np.ndarray of int
    """
    points_xy = np.stack((points.x, points.y), axis=-1).astype(np.float32)
    (xmin, ymin, xmax, ymax) = bounds

    non_empty_bin_keys, inverse, nb_pts_per_bin = np.unique(((points_xy - np.array([xmin, ymin]))
                                                             // ((xmax - xmin) / bin_per_dim + 0.000001)
                                                             ).astype(np.uint16),
                                                            axis=0, return_inverse=True, return_counts=True)

    idx_pts_bin = np.argsort(inverse)

    classification = np.zeros((len(points)), dtype=int)

    bin_slices = np.stack(
        [np.concatenate([[0], np.cumsum(nb_pts_per_bin)])[:-1], np.concatenate([[0], np.cumsum(nb_pts_per_bin)])[1:]],
        axis=-1)

    bin_args = [(points.z[idx_pts_bin[bin_slice[0]:bin_slice[1]]], idx_pts_bin[bin_slice[0]:bin_slice[1]],
                 bin_size, bin_range, first_bin_threshold, n_bins_after_first_bin)
                for bin_slice in bin_slices]

    # if num_workers is None:
    num_workers = os.cpu_count()
    with multiprocessing.Pool(num_workers) as pool:
        ground_idxs = pool.starmap(process_subtile, bin_args)
    ground_idx = np.concatenate(ground_idxs)

    classification[ground_idx] = True

    return classification


def sample_meshgrid(bounds, resolution):
    """
    Construct a numpy.meshgrid for regularly spaced points in the area enclosed in bounds.

    :param tuple or list or np.ndarray bounds: Structure containing the bounds of the area to sample in the xy-plane.
    :param int resolution: Int specifying the number of samples per direction.

    :return: Two 2D-arrays containing x- and y-position for each grid-cell respectively.
    :rtype: (np.ndarray, np.ndarray)
    """
    (xmin, ymin, xmax, ymax) = bounds

    x_steps = np.arange(xmin, xmax, (xmax - xmin) / resolution) + (xmax - xmin) / (2 * resolution)
    y_steps = np.arange(ymin, ymax, (ymax - ymin) / resolution) + (ymax - ymin) / (2 * resolution)

    xx, yy = np.meshgrid(x_steps, y_steps)

    return xx, yy


def grid_from_points(points, xx, yy, threshold=None):
    """
    Sample height and LiDAR-intensity of points along a grid specified by xx nd yy and apply a gaussian-filter to the
    result. Nearest neighbour interpolation is used for the sampling.

    :param LaserData points: Structured-array with the data to sample.
    :param np.ndarray xx: Array with the x-values of the sampling-points.
    :param np.ndarray yy: Array with the y-values of the sampling-points.
    :param float threshold: Optional. If provided returns a boolean-mask with "True" for all grid-cells within threshold
           distance to the nearest point.

    :return: Array of shape (xx.shape[0], xx.shape[1], 2) containing sampled heights and intensities.
             Optional: Boolean-mask masking grid-cells further than a given distance of the nearest point.
    :rtype: np.ndarray or (np.ndarray, np.ndarray of bool)
    """

    mask = None

    if len(points) > 0:
        points_xy = np.stack((points.x, points.y), axis=-1)
        values = np.stack((points.z, points["intensity"]), axis=-1)

        tree = cKDTree(points_xy)

        dists, idx = tree.query(np.c_[xx.ravel(), yy.ravel()], workers=-1)
        idx_mat = idx.reshape(xx.shape)
        array = values[idx_mat]
        if threshold is not None:
            dists_mat = dists.reshape(xx.shape)
            mask = dists_mat > threshold

    else:
        array = None

    if threshold is not None:
        return array, mask
    else:
        return array


def get_recording_info_from_shp(bounds, shapefile_dict):
    selection = shapefile_dict["file"].cx[bounds[0]:bounds[2], bounds[1]:bounds[3]]
    shape_dicts = selection.to_dict("records")
    return_dicts = [{"image_id": shape_dict[shapefile_dict["id_key"]],
                     "recordedAt": shape_dict[shapefile_dict["recording_time_key"]],
                     "recorderDirection": shape_dict[shapefile_dict["recording_direction_key"]],
                     "pos": [shape_dict["geometry"].x, shape_dict["geometry"].y],
                     "height": shape_dict[shapefile_dict["recording_height_key"]],
                     "image_path": shape_dict[shapefile_dict["image_path_key"]]
                     if shapefile_dict["image_path_key"] is not None else None,
                     "mask_path": shape_dict[shapefile_dict["mask_path_key"]]
                     if shapefile_dict["mask_path_key"] is not None else None,
                     "pitch": shape_dict[shapefile_dict["pitch_key"]]
                     if shapefile_dict["pitch_key"] is not None else 0,
                     "roll": shape_dict[shapefile_dict["roll_key"]]
                     if shapefile_dict["roll_key"] is not None else 0} for shape_dict in shape_dicts]

    return return_dicts


def get_image_infos_for_area(bounds, shapefile_dict, buffer=tile_buffer):
    """
    Get information for all panoramas within bounds.

    :param tuple or list or np.ndarray bounds: Structure containing the bounds of the area to get recordings for.
    :param float buffer: Float extending the area in all directions for buffer meters.
    :param dict or None shapefile_dict: Dictionary containing a shapefile with infos and the necessary keys.
    :return: A list of dictionaries with the relevant information for every recording within the requested area.
    :rtype: list of dict
    """
    (xmin, ymin, xmax, ymax) = bounds

    request_xmin = xmin - buffer
    request_xmax = xmax + buffer
    request_ymin = ymin - buffer
    request_ymax = ymax + buffer

    request_bounds = (request_xmin, request_ymin, request_xmax, request_ymax)

    return_dicts = get_recording_info_from_shp(request_bounds, shapefile_dict)

    return return_dicts


def cluster_infos_by_drive_lines(image_infos, base=36):
    """
    Cluster image info dictionaries returned by geosmartchange.cyclomedia.get_recording_info by continuous id's.

    :param list of dict image_infos: list of dictionaries containing information about panorama recordings.
    :param int base: base of the id
    :return: List of sublists containing infos. Each sublist represents one cluster.
    :rtype: list of (list of dict)
    """
    sorted_infos = sorted(image_infos, key=lambda info_: int(info_["image_id"], base=base))

    last_id_int = None

    clusters = []
    cluster = []

    for info in sorted_infos:
        id_int = int(info["image_id"], base=base)

        if last_id_int is None:
            cluster.append(info)
            last_id_int = id_int
            continue

        if id_int - last_id_int == 1:
            cluster.append(info)
            last_id_int = id_int
        else:
            clusters.append(cluster)
            cluster = [info]
            last_id_int = id_int

    clusters.append(cluster)

    return clusters


def combine_info_clusters(info_clusters, max_xy_dist_for_overlap=15, min_height_for_overlap=2):
    """
    Combine clusters of image infos if it is not possible they cross at different heights.

    :param list of (list of dict) info_clusters: List of clusters containing image infos.
    :param max_xy_dist_for_overlap: Maximal distance of two recording positions to be considered possible overlaps.
           If all infos in two clusters are further apart than this, they are automatically combined.
    :param min_height_for_overlap: Minimal height difference of two recording positions to be considered possible
           overlaps. If all infos in two clusters have less height difference, they are automatically combined.
    :return: list of combined clusters containing image infos. Number of combined clusters <= Number of input clusters.
    :rtype: list of (list of dict)
    """

    clusters = sorted(info_clusters, key=lambda _cluster: np.mean([info["height"] for info in _cluster]))

    if len(clusters) > 1:
        combined_clusters = []
        combined_cluster = None

        for cluster in clusters:
            if combined_cluster is None:
                combined_cluster = cluster
                continue

            combined_cluster_positions = np.array([info["pos"] for info in combined_cluster])
            cluster_positions = np.array([info["pos"] for info in cluster])

            flat_cluster_heights = np.array([info["height"] for info in combined_cluster])
            flatten_cluster_heights = np.array([info["height"] for info in cluster])

            xy_dists = cdist(combined_cluster_positions, cluster_positions)
            height_diffs = cdist(flat_cluster_heights.reshape((-1, 1)), flatten_cluster_heights.reshape((-1, 1)))

            if np.all(height_diffs[xy_dists < max_xy_dist_for_overlap] < min_height_for_overlap):
                combined_cluster = combined_cluster + cluster
            else:
                combined_clusters.append(combined_cluster)
                combined_cluster = cluster

        combined_clusters.append(combined_cluster)

    else:
        combined_clusters = clusters

    combined_clusters = sorted(combined_clusters, key=lambda cluster_: np.mean([info["height"] for info in cluster_]))

    return combined_clusters


def cluster_recording_infos(image_infos, max_xy_dist_for_overlap=15, min_height_for_overlap=2, base=36):
    """
    Cluster recording infos into height layers

    :param list of dict image_infos: list of dictionaries containing information about panorama recordings.
    :param max_xy_dist_for_overlap: Maximal distance of two recording positions to be considered possible overlaps.
           If all infos in two clusters are further apart than this, they are automatically combined.
    :param min_height_for_overlap: Minimal height difference of two recording positions to be considered possible
           overlaps. If all infos in two clusters have less height difference, they are automatically combined.
    :param int base: base of the id
    :return: list containing sublists representing the height layers and containing the info-dicts belonging to that
             layer.
    :rtype: list of (list of dict)
    """
    drive_line_clusters = cluster_infos_by_drive_lines(image_infos, base)
    clusters_n = combine_info_clusters(drive_line_clusters,
                                       max_xy_dist_for_overlap=max_xy_dist_for_overlap,
                                       min_height_for_overlap=min_height_for_overlap)

    return clusters_n


def cluster_laserdata_by_info_clusters(laserdata, info_clusters):
    """
    Cluster pointcloud according to clusters of info-dicts. Each recording point gets assigned the temporally nearest
    points of the pointcloud. Each cluster than gets assigned the points of all contained recording points.

    :param LaserData laserdata: LaserData object to cluster.
    :param list of (list of dict) info_clusters: Clusters of info-dicts, to cluster pointcloud by.
    :return: List of clustered pointclouds.
    :rtype: list of LaserData
    """
    recording_times = [[Time(isoparse(info["recordedAt"]), format="datetime", scale="tai").gps - 1e9 + 19]
                       for cluster in info_clusters for info in cluster]

    gps_times = laserdata["gps_time"]

    tree = cKDTree(recording_times)

    _, time_idx = tree.query(gps_times.reshape(-1, 1), workers=-1)

    laserdata_clusters = []
    processed_recordings = 0
    for i in range(len(info_clusters)):
        cluster_laserdata = laserdata[np.isin(time_idx, np.arange(len(info_clusters[i])) + processed_recordings)]
        laserdata_clusters.append(cluster_laserdata)
        processed_recordings += len(info_clusters[i])

    return laserdata_clusters


def load_image(image_path, image_type, image_id="Unknown"):
    if image_type == "jpeg":
        try:
            jpeg = TurboJPEG("/usr/lib/x86_64-linux-gnu/libturbojpeg.so.0")
        except OSError:
            jpeg = TurboJPEG()
        with open(image_path, 'rb') as img:
            image = img.read()
        try:
            image = jpeg.decode(image)
        except OSError:
            warnings.warn(f"File for image with id {image_id} could not be loaded correctly!")
            return None
    else:
        try:
            image = cv2.imread(image_path)
        except OSError:
            warnings.warn(f"File for image with id {image_id} could not be loaded correctly!")
            return None

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


class ImageLoader:
    """
    A class used to load images. Used to parallelize loading.
    """

    def __init__(self, images_dir, height=None, width=None, centering=None, car_mask=None):
        """
        :param str images_dir: Directory with the toplevel project-area folders.
        :param int height: Height to process the images with and return them in.
        :param int width: Width to process the images with and return them in.
        :param np.ndarray of np.uint8 or None car_mask: Optional. An uint8 image of the same shape as the images to load
               masking the recording car on the panoramas.
        """
        self.images_dir = images_dir
        self.car_mask = car_mask
        self.height = height
        self.width = width
        self.centering = centering or "Geographic"

    def __call__(self, image_info):
        """
        :param dict image_info: Dictionary containing information about the image to load and its recording position.
                                Must contain keys "image_id" and "recorderDirection".

        :return: The loaded and masked image.
        :rtype: np.ma.MaskedArray of np.uint8
        """
        if image_info.get("image_path") is not None:
            image_path = os.path.join(self.images_dir, image_info["image_path"])
        else:
            image_directory = os.path.join(self.images_dir, image_info["image_id"][:6])
            image_path = os.path.join(image_directory, f"{image_info['image_id']}.jpg")
        if not os.path.isfile(image_path):
            warnings.warn(f"Image with id {image_info['image_id']} not found at {image_path}!")
            return None
        image_type = imghdr.what(image_path)
        if image_type not in ["jpeg", "png"]:
            warnings.warn(f"Image with id {image_info['image_id']} is not a valid jpeg or png!")
            return None

        image = load_image(image_path, image_type, image_info['image_id'])
        if image is None:
            return None

        if self.height is not None and self.width is not None and (
                image.shape[0] != self.height or image.shape[1] != self.width):
            image = cv2.resize(image, (self.width, self.height))

        direction = image_info["recorderDirection"]

        mask = None
        if image_info.get("mask_path") is not None:
            mask_path = os.path.join(self.images_dir, image_info["mask_path"])
            if not os.path.isfile(mask_path):
                warnings.warn(f"No Mask found at {mask_path}!")
            else:
                mask_type = imghdr.what(mask_path)
                if mask_type not in ["jpeg", "png"]:
                    warnings.warn(f"Mask at {mask_path} is not a valid jpeg or png!")
                else:
                    mask = load_image(mask_path, mask_type, "mask")
                    if mask.shape[:2] != image.shape[:2]:
                        mask = cv2.resize(mask, (image.shape[1], image.shape[0]),
                                          interpolation=cv2.INTER_NEAREST)

        if mask is None:
            if self.car_mask is not None:
                if self.car_mask.shape[:2] != image.shape[:2]:
                    self.car_mask = cv2.resize(self.car_mask, (image.shape[1], image.shape[0]),
                                               interpolation=cv2.INTER_NEAREST)
                if "Recording_Direction" not in self.centering:
                    roll = int(direction / 360 * image.shape[1])
                    roll_mask = np.roll(self.car_mask, roll, axis=1)
                else:
                    roll_mask = self.car_mask
            else:
                roll_mask = np.zeros(image.shape)
                roll_mask[4625:, :] = [255, 255, 255]
        else:
            roll_mask = mask

        image = np.ma.masked_array(image, roll_mask.astype(bool))

        return image


def grid_convergence(point):
    """
    Get the grid-convergence angle between true-north and grid-north of the UTM-32-projection.

    :param np.ndarray or list or tuple point: Point to get grid-convergence for in UTM-32-coordinates.

    :return: Grid-convergence angle in radians.
    :rtype: float
    """
    central_meridian = 9
    long_lat_projector = pyproj.Proj(proj="utm", zone=32, ellps="WGS84")
    long, lat = long_lat_projector(point[0], point[1], inverse=True)

    long_rad = math.radians(long)
    lat_rad = math.radians(lat)
    cm_rad = math.radians(central_meridian)

    grid_convergence_angle = math.atan(math.tan(long_rad - cm_rad) * math.sin(lat_rad))

    return grid_convergence_angle


def create_and_assign_sample_mat(xx, yy, height_grid, recording_points, n_nearest=7):
    """
    Construct two ndarrays. One of shape (xx.shape[0], xx.shape[1], n_nearest, 3) containing positions to project into
    the images to sample, repeated n_nearest times along the second to last axis. The second one is of
    shape (xx.shape[0], xx.shape[1], n_nearest) and contains the corresponding image-indexes of the n_nearest images
    for every position.

    :param np.ndarray xx: Array with the x-values of the sampling-points.
    :param np.ndarray yy: Array with the y-values of the sampling-points.
    :param np.ndarray height_grid: Array with the heights of the sampling-points.
    :param np.ndarray recording_points: Array with the recording-position for all images.
    :param int n_nearest: Specifies the number of nearest images to sample per point.

    :return: Array of shape (xx.shape[0], xx,shape[1], n_nearest, 3) with repeated sampling positions and
             Array of shape (xx.shape[0], xx,shape[1], n_nearest) with the corresponding image indexes
    :rtype: (np.ndarray, np.ndarray of int)
    """
    n_nearest = min([n_nearest, len(recording_points)])

    recording_xys = recording_points[:, :2]

    tree = cKDTree(recording_xys)

    image_dists, image_idx = tree.query(np.c_[xx.ravel(), yy.ravel()], k=n_nearest, workers=-1)
    image_id_mat = np.array(image_idx).reshape((xx.shape[0], xx.shape[1], -1))

    xx_mat = np.repeat(xx[..., None], n_nearest, axis=-1)
    yy_mat = np.repeat(yy[..., None], n_nearest, axis=-1)
    height_mat = np.repeat(height_grid[..., None], n_nearest, axis=-1)
    point_mat = np.stack([xx_mat, yy_mat, height_mat], axis=-1)

    return point_mat, image_id_mat


def world_to_image_coordinates(point, recording_point, grid_convergence_angle, image_height, image_width,
                               inverse_rotation=None):
    """
    Transforms UTM-32-coordinates into panorama-coordinates.

    :param np.ndarray point: Array of shape (..., 3) with the UTM-32-coordinates of the points to get image-coordinates
           for. For just in time compilation this must have at least two dimensions eg. shape (1, 3).
    :param np.ndarray recording_point: Array of shape (..., 3) with the UTM-32-coordinates of the recording position of
           the image to project to. Can have fewer dimensions than point but dimensions must match the last dimensions
           of point.
    :param np.ndarray or float grid_convergence_angle: Array of arbitrary shape containing the grid-convergence angles
           to be used for the projection. Dimensions must match last dimensions of point.shape[:-1].
    :param int image_height: Height of the image to project to.
    :param int image_width: Width of the image to project to.
    :param np.ndarray or None inverse_rotation: inverse of the panoramas rotation in world space
    :return: An Array with the same shape as point except for the last dimension, witch is 2, containing the
             image-coordinates for point.
    :rtype: np.ndarray of int
    """
    delta = (point - recording_point).astype(np.float32)

    if inverse_rotation is not None:
        delta = delta[..., None, :] @ inverse_rotation
        delta = delta[..., 0, :]

    delta_x = delta[..., 0]
    delta_y = delta[..., 1]
    delta_h = delta[..., 2]

    beta_mat = np.arctan2(delta_x, delta_y) + np.pi + grid_convergence_angle
    alpha_mat = np.arctan2(np.sqrt(delta_x ** 2 + delta_y ** 2), delta_h)

    x_img_mat = beta_mat / (2 * np.pi) * image_width
    x_img_mat = x_img_mat.astype(np.int64) % image_width
    y_img_mat = (alpha_mat / np.pi) * image_height
    y_img_mat = y_img_mat.astype(np.int64)

    return np.stack((y_img_mat, x_img_mat), axis=-1)


def subsample_ortho(ortho, factor):
    assert ortho.shape[0] % factor == 0 and ortho.shape[1] % factor == 0

    return ortho[::factor, ::factor, ...]
