# @author: Maryniak, Marius - Fachbereich Elektrotechnik, Westfälische Hochschule Gelsenkirchen

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from shapely import LineString

from src.Planeness_Utils import (
    get_clustered_planeness_values_longitudinal,
    get_clustered_planeness_values_transverse)

pd.options.mode.chained_assignment = None


def preprocess_gdf(gdf):
    """
    | Returns the dataframe of the sampling points without the lane, index and geometry columns with
        the following schema:
    | cluster: id of the cluster
    | direction: id of the direction (0: longitudinal, 1: transverse)
    | z: height in meters

    :param gpd.GeoDataFrame gdf: geodataframe
    :returns: preprocessed dataframe
    :rtype: pd.DataFrame
    """
    return pd.DataFrame(gdf.drop(columns=['recording_id', 'height_layer', 'lane', 'index', 'geometry']))


def mask_df(df,
            column,
            value):
    """
    | Returns the masked dataframe.
    |
    | Based on:
    | https://stackoverflow.com/a/46165056

    :param pd.DataFrame or gpd.GeoDataFrame df: dataframe
    :param str column: name of the column
    :param str or int or float value: value
    :returns: masked dataframe
    :rtype: pd.DataFrame or gpd.GeoDataFrame
    """
    mask = df[column].values == value

    return df[mask]


def interpolate_heights(heights):
    """
    | Returns the interpolated heights.
    | NaN values are interpolated in transverse direction for each index.
    | The shape of the interpolated heights is (m, n) with m indices and n lanes.
    |
    | The shape of the heights is (m, n) with m indices and n lanes.

    :param np.ndarray[np.float64] heights: heights
    :returns: interpolated heights
    :rtype: np.ndarray[np.float64]
    """
    nan_indices_row = np.any(np.isnan(heights), axis=-1).nonzero()[0]

    for nan_index_row in nan_indices_row:
        non_nan_indices_column = np.where(~np.isnan(heights[nan_index_row]))[0]

        if non_nan_indices_column.shape[0] > 1:
            interpolate = interp1d(x=non_nan_indices_column,
                                   y=heights[nan_index_row, non_nan_indices_column],
                                   bounds_error=False,
                                   fill_value='extrapolate')

            heights[nan_index_row] = interpolate(np.arange(heights.shape[1]))

    return heights


def filter_curbs(heights,
                 curb_height=.04,
                 curb_distance=6):
    """
    | Returns the filtered heights.
    | Height values of curbs are replaced by NaN values.
    |
    | The shape of the heights is (n, 25) with n indices and 25 lanes.

    :param np.ndarray[np.float64] heights: heights
    :param float curb_height: height of the curb in meters
    :param int curb_distance: distance of the curb to the edge in sampling points
    :returns: filtered heights
    :rtype: np.ndarray[np.float64]
    """
    curbs_left_lane = np.diff(np.flip(heights[:, :curb_distance + 1], axis=-1)) >= curb_height  # last occurrence
    curb_indices_left_lane = curb_distance - np.where(curbs_left_lane.any(axis=-1),
                                                      np.argmax(curbs_left_lane, axis=-1),
                                                      curb_distance)

    curbs_right_lane = np.diff(heights[:, -curb_distance - 1:]) >= curb_height
    curb_indices_right_lane = np.where(curbs_right_lane.any(axis=-1),
                                       np.argmax(curbs_right_lane, axis=-1),
                                       curb_distance) + 24 - curb_distance

    mask = np.arange(25) < curb_indices_left_lane[:, np.newaxis]
    mask |= np.arange(25) > curb_indices_right_lane[:, np.newaxis]

    heights[mask] = np.nan

    return heights


def get_heights_longitudinal(df_longitudinal):
    """
    | Returns the longitudinal heights extracted from the dataframe of longitudinal heights.
    | NaN values are interpolated in longitudinal direction for each lane.
    | The shape of the longitudinal heights is (n, 5) with n indices and 5 lanes.

    :param pd.DataFrame df_longitudinal: dataframe of longitudinal heights
    :returns: longitudinal heights
    :rtype: np.ndarray[np.float64]
    """
    heights_longitudinal = np.array(df_longitudinal['z']).reshape(-1, 5)

    return interpolate_heights(heights_longitudinal.T).T


def get_heights_transverse(df_transverse):
    """
    | Returns the transverse heights extracted from the dataframe of transverse heights.
    | NaN values are interpolated in transverse and longitudinal direction for each index and lane in order
        to extrapolate indices that only contain NaN values.
    | Curbs are filtered by applying filter_curbs() to the transverse heights.
    | The shape of the transverse heights is (n, 25) with n indices and 25 lanes.

    :param pd.DataFrame df_transverse: dataframe of transverse heights
    :returns: transverse heights
    :rtype: np.ndarray[np.float64]
    """
    heights_transverse = np.array(df_transverse['z']).reshape(-1, 25)
    heights_transverse = interpolate_heights(heights_transverse)
    heights_transverse = interpolate_heights(heights_transverse.T).T
    heights_transverse = filter_curbs(heights_transverse)

    return interpolate_heights(heights_transverse)


def get_clustered_heights_longitudinal(df):
    """
    | Applies get_heights_longitudinal() to each cluster of longitudinal heights.
    | Returns the clustered longitudinal heights extracted from the dataframe.
    | The shape of the longitudinal heights of each cluster is (n, 5) with n indices and 5 lanes.

    :param pd.DataFrame df: dataframe
    :returns: clustered longitudinal heights
    :rtype: list[np.ndarray[np.float64]]
    """
    df_longitudinal = mask_df(df,
                              column='direction',
                              value=0)
    clustered_df_longitudinal = df_longitudinal.groupby('cluster')

    return [get_heights_longitudinal(df_longitudinal) for _, df_longitudinal in clustered_df_longitudinal]


def get_clustered_heights_transverse(df):
    """
    | Applies get_heights_transverse() to each cluster of transverse heights.
    | Returns the clustered transverse heights extracted from the dataframe.
    | The shape of the transverse heights of each cluster is (n, 25) with n indices and 25 lanes.

    :param pd.DataFrame df: dataframe
    :returns: clustered transverse heights
    :rtype: list[np.ndarray[np.float64]]
    """
    df_transverse = mask_df(df,
                            column='direction',
                            value=1)
    clustered_df_transverse = df_transverse.groupby('cluster')

    return [get_heights_transverse(df_transverse) for _, df_transverse in clustered_df_transverse]


def get_gdf_longitudinal(gdf,
                         planeness_values_longitudinal,
                         crs='EPSG:25832'):
    """
    | Returns the geodataframe of the longitudinal planeness values with the following schema:
    | cluster: id of the cluster
    | lane: id of the lane (0: left_2, 1: left_1, 2: mid, 3: right_1, 4: right_2)
    | index: id of the index along the lanes
    | depth_mid: distance between the rolling straight edge and the height in meters
    | delta_depth_mid_moving_average_11: moving average of depth_mid with a window size of 11 sampling points (1m)
        in meters
    | delta_depth_mid_moving_average_31: moving average of depth_mid with a window size of 31 sampling points (3m)
        in meters
    | delta_depth_mid_moving_average_101: moving average of depth_mid with a window size of 101 sampling points (10m)
        in meters
    | delta_depth_mid_moving_average_301: moving average of depth_mid with a window size of 301 sampling points (30m)
        in meters
    | gradient: gradient of the rolling straight edge in percent
    | geometry: x and y coordinates of the point
    |
    | The shape of the longitudinal planeness values is (n, 6) with n sampling points and 6 planeness values
        (depth_mid, delta_depth_mid_moving_average_11, delta_depth_mid_moving_average_31,
        delta_depth_mid_moving_average_101, delta_depth_mid_moving_average_301, gradient).

    :param gpd.GeoDataFrame gdf: geodataframe
    :param np.ndarray[np.float64] planeness_values_longitudinal: longitudinal planeness values
    :param str crs: coordinate reference system
    :returns: geodataframe of longitudinal planeness values
    :rtype: gpd.GeoDataFrame
    """
    gdf_longitudinal = mask_df(gdf,
                               column='direction',
                               value=0)

    gdf_longitudinal.reset_index(drop=True, inplace=True)
    gdf_longitudinal.drop(columns=['direction', 'z'], inplace=True)

    gdf_longitudinal.insert(3, 'depth_mid', planeness_values_longitudinal[:, 0])
    gdf_longitudinal.insert(4, 'delta_depth_mid_moving_average_11', planeness_values_longitudinal[:, 1])
    gdf_longitudinal.insert(5, 'delta_depth_mid_moving_average_31', planeness_values_longitudinal[:, 2])
    gdf_longitudinal.insert(6, 'delta_depth_mid_moving_average_101', planeness_values_longitudinal[:, 3])
    gdf_longitudinal.insert(7, 'delta_depth_mid_moving_average_301', planeness_values_longitudinal[:, 4])
    gdf_longitudinal.insert(8, 'gradient', planeness_values_longitudinal[:, 5])

    gdf_longitudinal.set_crs(crs, inplace=True)

    return gdf_longitudinal


def get_gdf_transverse(gdf,
                       planeness_values_transverse,
                       crs='EPSG:25832'):
    """
    | Returns the geodataframe of the transverse planeness values with the following schema:
    | cluster: id of the cluster
    | index: id of the index along the lane
    | rut_depth_left: depth of the left rut in meters
    | rut_depth_right: depth of the right rut in meters
    | water_depth_left: water depth in the left rut in meters
    | water_depth_right: water depth in the right rut in meters
    | gradient: gradient of the regression line in percent (note that the sign is flipped)
    | geometry: x and y coordinates of the line
    |
    | The shape of the transverse planeness values is (n, 5) with n lines and 5 planeness values
        (rut_depth_left, rut_depth_right, water_depth_left, water_depth_right, gradient).

    :param gpd.GeoDataFrame gdf: geodataframe
    :param np.ndarray[np.float64] planeness_values_transverse: transverse planeness values
    :param str crs: coordinate reference system
    :returns: geodataframe of transverse planeness values
    :rtype: gpd.GeoDataFrame
    """
    gdf_transverse = mask_df(gdf,
                             column='direction',
                             value=1)

    gdf_transverse.reset_index(drop=True, inplace=True)
    gdf_transverse.drop(columns=['direction', 'lane', 'z'], inplace=True)

    indices = np.arange(0, gdf_transverse.shape[0], 25)
    indices = np.repeat(indices, 2)
    indices[1::2] += 24

    gdf_transverse = gdf_transverse.iloc[indices]

    gdf_transverse['next_geometry'] = gdf_transverse.geometry.shift(-1)
    gdf_transverse = gdf_transverse[:-1]

    gdf_transverse['line'] = gdf_transverse.apply(lambda row: LineString([row['geometry'], row['next_geometry']]),
                                                  axis=1)
    gdf_transverse = gdf_transverse[::2]

    gdf_transverse.reset_index(drop=True, inplace=True)
    gdf_transverse.drop(columns=['geometry', 'next_geometry'], inplace=True)
    gdf_transverse.rename(columns={'line': 'geometry'}, inplace=True)
    gdf_transverse.set_geometry('geometry', inplace=True)

    gdf_transverse.insert(2, 'rut_depth_left', planeness_values_transverse[:, 0])
    gdf_transverse.insert(3, 'rut_depth_right', planeness_values_transverse[:, 1])
    gdf_transverse.insert(4, 'water_depth_left', planeness_values_transverse[:, 2])
    gdf_transverse.insert(5, 'water_depth_right', planeness_values_transverse[:, 3])
    gdf_transverse.insert(6, 'gradient', planeness_values_transverse[:, 4])

    gdf_transverse.set_crs(crs, inplace=True)

    return gdf_transverse


def aggregate_gdf(gdf_aggregation_areas,
                  gdf_longitudinal,
                  gdf_transverse):
    """
    | Returns the aggregated geodataframe with appended statistical values for each polygon with the following schema:
    | original schema
    | depth_mid_max: maximum of depth_mid in meters
    | depth_mid_mean: mean of depth_mid in meters
    | delta_depth_mid_moving_average_11_std: standard deviation of delta_depth_mid_moving_average_11 in meters
    | delta_depth_mid_moving_average_31_std: standard deviation of delta_depth_mid_moving_average_31 in meters
    | delta_depth_mid_moving_average_101_std: standard deviation of delta_depth_mid_moving_average_101 in meters
    | delta_depth_mid_moving_average_301_std: standard deviation of delta_depth_mid_moving_average_301 in meters
    | gradient_longitudinal_mean: mean of gradient_longitudinal in percent
    | rut_depth_left_max: maximum of rut_depth_left in meters
    | rut_depth_right_max: maximum of rut_depth_right in meters
    | rut_depth_left_mean: mean of rut_depth_left in meters
    | rut_depth_right_mean: mean of rut_depth_right in meters
    | rut_depth_mean_max: maximum of rut_depth_left_mean and rut_depth_right_mean in meters
    | rut_depth_max: maximum of rut_depth_left_max and rut_depth_right_max in meters
    | rut_depth_left_std: standard deviation of rut_depth_left in meters
    | rut_depth_right_std: standard deviation of rut_depth_right in meters
    | water_depth_left_max: maximum of water_depth_left in meters
    | water_depth_right_max: maximum of water_depth_right in meters
    | water_depth_left_mean: mean of water_depth_left in meters
    | water_depth_right_mean: mean of water_depth_right in meters
    | water_depth_mean_max: maximum of water_depth_left_mean and water_depth_right_mean in meters
    | water_depth_max: maximum of water_depth_left_max and water_depth_right_max in meters
    | water_depth_left_std: standard deviation of water_depth_left in meters
    | water_depth_right_std: standard deviation of water_depth_right in meters
    | gradient_transverse_mean: mean of gradient_transverse in percent (note that the sign is flipped)

    :param gpd.GeoDataFrame gdf_aggregation_areas: geodataframe with polygons to aggregate the planeness values to
    :param gpd.GeoDataFrame gdf_longitudinal: geodataframe of longitudinal planeness values
    :param gpd.GeoDataFrame gdf_transverse: geodataframe of transverse planeness values
    :returns: aggregated geodataframe
    :rtype: gpd.GeoDataFrame
    """
    gdf = gdf_aggregation_areas

    gdf_aggregated_longitudinal = gpd.sjoin(left_df=gdf_longitudinal,
                                            right_df=gdf,
                                            how='inner',
                                            predicate='within')

    depth_mid_max = gdf_aggregated_longitudinal.groupby('index_right')['depth_mid'].max()
    gdf['depth_mid_max'] = gdf.index.map(depth_mid_max)

    depth_mid_mean = gdf_aggregated_longitudinal.groupby('index_right')['depth_mid'].mean()
    gdf['depth_mid_mean'] = gdf.index.map(depth_mid_mean)

    delta_depth_mid_moving_average_11_std = \
        gdf_aggregated_longitudinal.groupby('index_right')['delta_depth_mid_moving_average_11'].std(ddof=0)
    gdf['delta_depth_mid_moving_average_11_std'] = gdf.index.map(delta_depth_mid_moving_average_11_std)

    delta_depth_mid_moving_average_31_std = \
        gdf_aggregated_longitudinal.groupby('index_right')['delta_depth_mid_moving_average_31'].std(ddof=0)
    gdf['delta_depth_mid_moving_average_31_std'] = gdf.index.map(delta_depth_mid_moving_average_31_std)

    delta_depth_mid_moving_average_101_std = \
        gdf_aggregated_longitudinal.groupby('index_right')['delta_depth_mid_moving_average_101'].std(ddof=0)
    gdf['delta_depth_mid_moving_average_101_std'] = gdf.index.map(delta_depth_mid_moving_average_101_std)

    delta_depth_mid_moving_average_301_std = \
        gdf_aggregated_longitudinal.groupby('index_right')['delta_depth_mid_moving_average_301'].std(ddof=0)
    gdf['delta_depth_mid_moving_average_301_std'] = gdf.index.map(delta_depth_mid_moving_average_301_std)

    gradient_longitudinal_mean = gdf_aggregated_longitudinal.groupby('index_right')['gradient'].mean()
    gdf['gradient_longitudinal_mean'] = gdf.index.map(gradient_longitudinal_mean)

    gdf_aggregated_transverse = gpd.sjoin(left_df=gdf_transverse,
                                          right_df=gdf,
                                          how='inner',
                                          predicate='intersects')

    rut_depth_left_max = gdf_aggregated_transverse.groupby('index_right')['rut_depth_left'].max()
    gdf['rut_depth_left_max'] = gdf.index.map(rut_depth_left_max)

    rut_depth_right_max = gdf_aggregated_transverse.groupby('index_right')['rut_depth_right'].max()
    gdf['rut_depth_right_max'] = gdf.index.map(rut_depth_right_max)

    rut_depth_left_mean = gdf_aggregated_transverse.groupby('index_right')['rut_depth_left'].mean()
    gdf['rut_depth_left_mean'] = gdf.index.map(rut_depth_left_mean)

    rut_depth_right_mean = gdf_aggregated_transverse.groupby('index_right')['rut_depth_right'].mean()
    gdf['rut_depth_right_mean'] = gdf.index.map(rut_depth_right_mean)

    gdf['rut_depth_mean_max'] = gdf[['rut_depth_left_mean', 'rut_depth_right_mean']].max(axis=1)
    gdf['rut_depth_max'] = gdf[['rut_depth_left_max', 'rut_depth_right_max']].max(axis=1)

    rut_depth_left_std = gdf_aggregated_transverse.groupby('index_right')['rut_depth_left'].std(ddof=0)
    gdf['rut_depth_left_std'] = gdf.index.map(rut_depth_left_std)

    rut_depth_right_std = gdf_aggregated_transverse.groupby('index_right')['rut_depth_right'].std(ddof=0)
    gdf['rut_depth_right_std'] = gdf.index.map(rut_depth_right_std)

    water_depth_left_max = gdf_aggregated_transverse.groupby('index_right')['water_depth_left'].max()
    gdf['water_depth_left_max'] = gdf.index.map(water_depth_left_max)

    water_depth_right_max = gdf_aggregated_transverse.groupby('index_right')['water_depth_right'].max()
    gdf['water_depth_right_max'] = gdf.index.map(water_depth_right_max)

    water_depth_left_mean = gdf_aggregated_transverse.groupby('index_right')['water_depth_left'].mean()
    gdf['water_depth_left_mean'] = gdf.index.map(water_depth_left_mean)

    water_depth_right_mean = gdf_aggregated_transverse.groupby('index_right')['water_depth_right'].mean()
    gdf['water_depth_right_mean'] = gdf.index.map(water_depth_right_mean)

    gdf['water_depth_mean_max'] = gdf[['water_depth_left_mean', 'water_depth_right_mean']].max(axis=1)
    gdf['water_depth_max'] = gdf[['water_depth_left_max', 'water_depth_right_max']].max(axis=1)

    water_depth_left_std = gdf_aggregated_transverse.groupby('index_right')['water_depth_left'].std(ddof=0)
    gdf['water_depth_left_std'] = gdf.index.map(water_depth_left_std)

    water_depth_right_std = gdf_aggregated_transverse.groupby('index_right')['water_depth_right'].std(ddof=0)
    gdf['water_depth_right_std'] = gdf.index.map(water_depth_right_std)

    gradient_transverse_mean = gdf_aggregated_transverse.groupby('index_right')['gradient'].mean()
    gdf['gradient_transverse_mean'] = gdf.index.map(gradient_transverse_mean)
    return gdf


def get_ap9_compliant_gdf(gdf,
                          precision=2,
                          drop_original_schema=False):
    """
    | Returns the AP9 compliant aggregated geodataframe with appended statistical values for each polygon with
        the following schema (according to 'Arbeitspapier 9 K zur Systematik der Straßenerhaltung'):
    | original schema
    | SM4L_M: Maximum der Stichmaße in Lattenmitte unter der 4m Latte [mm]
    | SM4L_A: Mittelwert der Stichmaße in Lattenmitte unter der 4m Latte [mm]
    | S01: Standardabweichung der Differenzen zum gleitenden Mittelwert bei einer Mittelungslänge von 1m [mm]
    | S03: Standardabweichung der Differenzen zum gleitenden Mittelwert bei einer Mittelungslänge von 3m [mm]
    | S10: Standardabweichung der Differenzen zum gleitenden Mittelwert bei einer Mittelungslänge von 10m [mm]
    | S30: Standardabweichung der Differenzen zum gleitenden Mittelwert bei einer Mittelungslänge von 30m [mm]
    | LN: Mittelwert der Längsneigungen [%]
    | MSPTL: Mittelwert der linken Spurrinnentiefen nach dem 1,2m Latten-Prinzip [mm]
    | MSPTR: Mittelwert der rechten Spurrinnentiefen nach dem 1,2m Latten-Prinzip [mm]
    | MSPT: Maximum der Mittelwerte der linken und rechten Spurrinnentiefen [mm]
    | SPTMAX: Maximum der Spurrinnentiefen [mm]
    | SSPTL: Standardabweichung der linken Spurrinnentiefen [mm]
    | SSPTR: Standardabweichung der rechten Spurrinnentiefen [mm]
    | MSPHL: Mittelwert der linken fiktiven Wassertiefen [mm]
    | MSPHR: Mittelwert der rechten fiktiven Wassertiefen [mm]
    | MSPH: Maximum der Mittelwerte der linken und rechten fiktiven Wassertiefen [mm]
    | SSPHL: Standardabweichung der linken fiktiven Wassertiefen [mm]
    | SSPHR: Standardabweichung der rechten fiktiven Wassertiefen [mm]
    | QN: Mittelwert der Querneigungen [%]

    :param gpd.GeoDataFrame gdf: aggregated geodataframe
    :param int precision: precision of the float values
    :param bool drop_original_schema: if True, the original schema and their attributes are dropped
    :returns: aggregated geodataframe
    :rtype: gpd.GeoDataFrame
    """
    ap9_compliant_schema = {'depth_mid_max': 'SM4L_M',
                            'depth_mid_mean': 'SM4L_A',
                            'delta_depth_mid_moving_average_11_std': 'S01',
                            'delta_depth_mid_moving_average_31_std': 'S03',
                            'delta_depth_mid_moving_average_101_std': 'S10',
                            'delta_depth_mid_moving_average_301_std': 'S30',
                            'gradient_longitudinal_mean': 'LN',
                            'rut_depth_left_mean': 'MSPTL',
                            'rut_depth_right_mean': 'MSPTR',
                            'rut_depth_mean_max': 'MSPT',
                            'rut_depth_max': 'SPTMAX',
                            'rut_depth_left_std': 'SSPTL',
                            'rut_depth_right_std': 'SSPTR',
                            'water_depth_left_mean': 'MSPHL',
                            'water_depth_right_mean': 'MSPHR',
                            'water_depth_mean_max': 'MSPH',
                            'water_depth_left_std': 'SSPHL',
                            'water_depth_right_std': 'SSPHR',
                            'gradient_transverse_mean': 'QN'}

    columns_internal_use = ['aggregation_area_id',
                            'rut_depth_left_max',
                            'rut_depth_right_max',
                            'water_depth_left_max',
                            'water_depth_right_max',
                            'water_depth_max']

    if drop_original_schema:
        gdf.drop(columns=gdf.columns.difference(['geometry'] + [*ap9_compliant_schema]), inplace=True)
    else:
        gdf.drop(columns=columns_internal_use, inplace=True)

    gdf[[column for column in [*ap9_compliant_schema] if 'gradient' not in column]] *= 1e3

    for column in gdf.select_dtypes(include=['float64']).columns:
        gdf[column] = gdf[column].apply(lambda x: round(x, precision) if not pd.isnull(x) else x)

    gdf.rename(columns=ap9_compliant_schema, inplace=True)
    return gdf


def process_gdf_planeness(gdf_planeness,
                          gdf_aggregation_areas,
                          crs='EPSG:25832'):
    """
    | Returns the AP9 compliant aggregated geodataframe with appended statistical values for each polygon with
        the following schema (according to 'Arbeitspapier 9 K zur Systematik der Straßenerhaltung'):
    | original schema
    | SM4L_M: Maximum der Stichmaße in Lattenmitte unter der 4m Latte [mm]
    | SM4L_A: Mittelwert der Stichmaße in Lattenmitte unter der 4m Latte [mm]
    | S01: Standardabweichung der Differenzen zum gleitenden Mittelwert bei einer Mittelungslänge von 1m [mm]
    | S03: Standardabweichung der Differenzen zum gleitenden Mittelwert bei einer Mittelungslänge von 3m [mm]
    | S10: Standardabweichung der Differenzen zum gleitenden Mittelwert bei einer Mittelungslänge von 10m [mm]
    | S30: Standardabweichung der Differenzen zum gleitenden Mittelwert bei einer Mittelungslänge von 30m [mm]
    | LN: Mittelwert der Längsneigungen [%]
    | MSPTL: Mittelwert der linken Spurrinnentiefen nach dem 1,2m Latten-Prinzip [mm]
    | MSPTR: Mittelwert der rechten Spurrinnentiefen nach dem 1,2m Latten-Prinzip [mm]
    | MSPT: Maximum der Mittelwerte der linken und rechten Spurrinnentiefen [mm]
    | SPTMAX: Maximum der Spurrinnentiefen [mm]
    | SSPTL: Standardabweichung der linken Spurrinnentiefen [mm]
    | SSPTR: Standardabweichung der rechten Spurrinnentiefen [mm]
    | MSPHL: Mittelwert der linken fiktiven Wassertiefen [mm]
    | MSPHR: Mittelwert der rechten fiktiven Wassertiefen [mm]
    | MSPH: Maximum der Mittelwerte der linken und rechten fiktiven Wassertiefen [mm]
    | SSPHL: Standardabweichung der linken fiktiven Wassertiefen [mm]
    | SSPHR: Standardabweichung der rechten fiktiven Wassertiefen [mm]
    | QN: Mittelwert der Querneigungen [%]

    :param gpd.GeoDataFrame gdf_planeness: geodataframe of planeness values
    :param gpd.GeoDataFrame gdf_aggregation_areas: geodataframe with polygons to aggregate the planeness values to
    :param str crs: coordinate reference system
    :returns: aggregated geodataframe
    :rtype: gpd.GeoDataFrame
    """
    df = preprocess_gdf(gdf=gdf_planeness)

    clustered_heights_longitudinal = get_clustered_heights_longitudinal(df=df)
    clustered_heights_transverse = get_clustered_heights_transverse(df=df)

    clustered_planeness_values_longitudinal = get_clustered_planeness_values_longitudinal(
        clustered_heights=clustered_heights_longitudinal)
    clustered_planeness_values_transverse = get_clustered_planeness_values_transverse(
        clustered_heights=clustered_heights_transverse)

    gdf_longitudinal = get_gdf_longitudinal(gdf=gdf_planeness,
                                            planeness_values_longitudinal=clustered_planeness_values_longitudinal,
                                            crs=crs)
    gdf_transverse = get_gdf_transverse(gdf=gdf_planeness,
                                        planeness_values_transverse=clustered_planeness_values_transverse,
                                        crs=crs)

    gdf_aggregated = aggregate_gdf(gdf_aggregation_areas=gdf_aggregation_areas,
                                   gdf_longitudinal=gdf_longitudinal,
                                   gdf_transverse=gdf_transverse)

    gdf_aggregated = get_ap9_compliant_gdf(gdf=gdf_aggregated)

    columns = ['SM4L_M', 'SM4L_A', 'S01', 'S03', 'S10', 'S30', 'LN', 'MSPTL', 'MSPTR', 'MSPT', 'SPTMAX',
               'SSPTL', 'SSPTR', 'MSPHL', 'MSPHR', 'MSPH', 'SSPHL', 'SSPHR', 'QN']

    gdf_aggregated.loc[gdf_aggregated['FK'] == 'N', columns] = np.nan

    return gdf_aggregated
