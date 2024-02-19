# @author: Roß, Alexander - Fachbereich Elektrotechnik, Westfälische Hochschule Gelsenkirchen
# @coauthor: Maryniak, Marius - Fachbereich Elektrotechnik, Westfälische Hochschule Gelsenkirchen

from collections import OrderedDict

import geopandas as gpd
import numpy as np
import pandas as pd
from rasterio import features
from rasterio.transform import Affine, from_origin
from shapely.geometry import shape

# Ignore division by 0 warning.
np.seterr(divide="ignore", invalid="ignore")

# Colors for saving condition raster map.
condition_colors = np.array([(0, 0, 0), (35, 235, 35), (235, 235, 35), (235, 35, 35)])

# Colors for saving type raster map.
type_colors = np.array([(0, 0, 0), (35, 135, 35), (35, 235, 35), (35, 35, 135), (35, 35, 235), (135, 35, 135),
                        (35, 135, 135), (235, 235, 35), (235, 135, 35), (135, 35, 35)])

# Field names in shapefile for types and associated percentages.
type_field_names = [
    ["duenn_Antl", "duenn_gut", "duenn_mtl", "duenn_schl"],
    ["dnGst_Antl", "dnGst_gut", "dnGst_mtl", "dnGst_schl"],
    ["brt_Antl", "brt_gut", "brt_mtl", "brt_schl"],
    ["brGst_Antl", "brGst_gut", "brGst_mtl", "brGst_schl"],
    ["exBrt_Antl", "exBrt_gut", "exBrt_mtl", "exBrt_schl"],
    ["ignore", "ignore", "ignore", "ignore"],
    ["pfeil_Antl", "pfeil_gut", "pfeil_mtl", "pfeil_schl"],
    ["symbl_Antl", "symbl_gut", "symbl_mtl", "symbl_schl"],
    ["sperr_Antl", "sperr_gut", "sperr_mtl", "sperr_schl"]
]

# Datatypes for shapefile
str80 = "str:80"
float3_2 = "float:3.2"

# Schema for markings shapefile
object_schema = {'properties': OrderedDict([('typ', str80),
                                            ('zustand', str80),
                                            ('gut', float3_2),
                                            ('mittel', float3_2),
                                            ('schlecht', float3_2),
                                            ('ebene', str80)]),
                 'geometry': 'Polygon'}

# Schema for tiles shapefile
tile_schema = {'properties': OrderedDict([('FileName', str80),
                                          ('ebene', str80),
                                          ('Zustand', str80),
                                          ('gut', float3_2),
                                          ('mittel', float3_2),
                                          ('schlecht', float3_2),
                                          ('Mark_Antl', float3_2),
                                          ('duenn_Antl', float3_2),
                                          ('duenn_gut', float3_2),
                                          ('duenn_mtl', float3_2),
                                          ('duenn_schl', float3_2),
                                          ('dnGst_Antl', float3_2),
                                          ('dnGst_gut', float3_2),
                                          ('dnGst_mtl', float3_2),
                                          ('dnGst_schl', float3_2),
                                          ('brt_Antl', float3_2),
                                          ('brt_gut', float3_2),
                                          ('brt_mtl', float3_2),
                                          ('brt_schl', float3_2),
                                          ('brGst_Antl', float3_2),
                                          ('brGst_gut', float3_2),
                                          ('brGst_mtl', float3_2),
                                          ('brGst_schl', float3_2),
                                          ('exBrt_Antl', float3_2),
                                          ('exBrt_gut', float3_2),
                                          ('exBrt_mtl', float3_2),
                                          ('exBrt_schl', float3_2),
                                          ('pfeil_Antl', float3_2),
                                          ('pfeil_gut', float3_2),
                                          ('pfeil_mtl', float3_2),
                                          ('pfeil_schl', float3_2),
                                          ('symbl_Antl', float3_2),
                                          ('symbl_gut', float3_2),
                                          ('symbl_mtl', float3_2),
                                          ('symbl_schl', float3_2),
                                          ('sperr_Antl', float3_2),
                                          ('sperr_gut', float3_2),
                                          ('sperr_mtl', float3_2),
                                          ('sperr_schl', float3_2)]),
               'geometry': 'Polygon'}

substance_types = ['r', 'onf', 'af', 'aa']


def classify_marking_dataframe(dataframe, thresholds):
    """
    Classifies dataframe containing markings by given thresholds.

    :param gpd.GeoDataFrame dataframe: Dataframe containing individual markings.
    :param dict thresholds: Dict containing thresholds for weighting.
    :return: Reweighed dataframe.
    :rtype: gpd.GeoDataFrame
    """
    percent_mittel = dataframe["mittel"] / 100
    percent_schlecht = dataframe["schlecht"] / 100

    result = np.array(["gut"] * len(dataframe), dtype="object")
    result[percent_mittel + percent_schlecht >= thresholds["markings_mittel"]] = "mittel"
    result[percent_schlecht >= thresholds["markings_schlecht"]] = "schlecht"

    dataframe["zustand"] = result

    return dataframe


def get_roadmarkings_gdf(condition_mask, type_mask, file_offset, file_res, thresholds, height_layer=0,
                         crs="epsg:25832"):
    """
    Creates shapefile with marking-objects from masks and positioning info.

    :param np.ndarray condition_mask: Mask containing condition classes.
    :param np.ndarray type_mask: Mask containing type classes.
    :param tuple of int or np.ndarray of int or list of int file_offset: Coordinates of the masks south-west corner
        (easting, northing).
    :param float file_res: Resolution of the masks in m / pixel.
    :param dict thresholds: Dict containing thresholds for classifying the resulting shapes.
    :param int height_layer: Hight-layer the masks belong to.
    :param str crs: EPGS-code to create shapes in. Must match other coordinates given.
    :return: Dataframe representing the generated shapes.
    :rtype: gpd.GeoDataFrame
    """
    types = ["Linie dünn", "Linie dünn gestrichelt", "Linie breit", "Linie breit gestrichelt", "Linie extrabreit",
             "NULL", "Pfeile", "Symbole", "Sperrflächen"]

    condition_mask = condition_mask[::-1]
    type_mask = type_mask[::-1]

    combined_type_mask = type_mask.copy()

    # --- This will make very good predictions worse. Remove if model is sufficiently good.
    combined_type_mask[combined_type_mask == 2] = 1
    combined_type_mask[combined_type_mask == 4] = 3
    # ---

    shape_dicts = []

    mask_type_mask = combined_type_mask != 0

    transform = Affine.translation(file_offset[0], file_offset[1]) * Affine.scale(file_res, file_res)

    shapes = features.shapes(combined_type_mask, mask=mask_type_mask, connectivity=8, transform=transform)

    for _shp, value in shapes:

        shape_geom = shape(_shp)
        shape_mask = features.rasterize((shape_geom, 1), out_shape=condition_mask.shape, transform=transform)

        type_shape_pixels = type_mask[shape_mask.astype(bool)]

        type_counts = np.zeros((len(types) + 1))

        for type_id, counts in np.stack(np.unique(type_shape_pixels, return_counts=True), axis=-1):
            type_counts[type_id] = counts

        shape_type = types[int(np.argmax(type_counts)) - 1]

        shape_pixels = condition_mask[shape_mask.astype(bool)]
        num_nonzero = np.count_nonzero(shape_pixels)
        if num_nonzero == 0:
            continue
        num_gut = np.count_nonzero(shape_pixels == 1)
        num_mittel = np.count_nonzero(shape_pixels == 2)
        num_schlecht = np.count_nonzero(shape_pixels == 3)

        percent_gut = 100 * num_gut / num_nonzero
        percent_mittel = 100 * num_mittel / num_nonzero
        percent_schlecht = 100 * num_schlecht / num_nonzero

        shape_dict = {"typ": shape_type,
                      "zustand": np.nan,
                      "gut": percent_gut,
                      "mittel": percent_mittel,
                      "schlecht": percent_schlecht,
                      "ebene": str(height_layer),
                      "geometry": shape_geom}
        shape_dicts.append(shape_dict)

    if shape_dicts:
        output_shp = gpd.GeoDataFrame(shape_dicts, crs=crs)
        output_shp = classify_marking_dataframe(output_shp, thresholds)
    else:
        output_shp = gpd.GeoDataFrame(geometry=[], crs=crs)

    return output_shp


def get_substances_gdf(substance_mask, file_offset, file_res, height_layer=0, crs="epsg:25832"):

    transform = from_origin(west=file_offset[0],
                            north=file_offset[1] + file_res * substance_mask.shape[1],
                            xsize=file_res,
                            ysize=file_res)

    gdfs = []

    for i, class_name in enumerate(substance_types):
        vectorized_mask = features.shapes(substance_mask[..., i], transform=transform)
        polygons = [{"properties": {"typ": class_name, "ebene": height_layer}, "geometry": polygon}
                    for polygon, value in vectorized_mask if int(value) != 0]

        if len(polygons):
            gdf = gpd.GeoDataFrame.from_features(polygons, crs=crs)
            gdfs.append(gdf)

    if gdfs:
        gdf = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs=crs)
    else:
        gdf = gpd.GeoDataFrame(geometry=[], crs=crs)

    return gdf


def assign_percentages_types(nums_types, area_dict):
    """
    Assigns percentages for types and type-condition combinations to dataframe containing tiles.

    :param np.ndarray or list of list nums_types: Array of same shape as type_field_names containing absolute number of
           pixels for each entry.
    :param dict area_dict: Dict to assign percentages to.
    :return: Dict containing tiles, filled with percentages.
    :rtype: dict
    """
    num_any_total = np.sum(np.array(nums_types)[:, 0])

    for j, field_names in enumerate(type_field_names):
        for k, field_name in enumerate(field_names):
            if field_name != "ignore":

                if "Antl" in field_name:
                    value = 100 * nums_types[j][k] / num_any_total
                elif nums_types[j][0] == 0:
                    value = np.nan
                else:
                    value = 100 * nums_types[j][k] / nums_types[j][0]
                area_dict[field_name] = value

    return area_dict


def classify_tile_dataframe(dataframe, factors, thresholds):
    """
    Classifies dataframe containing tiles with given factors and thresholds.

    :param gpd.GeoDataFrame dataframe: Dataframe containing tiles.
    :param np.ndarray factors: Array with weighting factors for each type-condition combination.
    :param dict thresholds: Dict containing thresholds for classification.
    :return:
    """
    factors = np.repeat(factors[..., np.newaxis], len(dataframe), axis=-1)
    percentages_types = []
    for field_names in type_field_names:
        percentages_type = []
        for field_name in field_names:
            percentages_type.append(np.array(dataframe[field_name], dtype=float) if field_name != "ignore"
                                    else np.zeros(len(dataframe)))
        percentages_types.append(percentages_type)

    percentages_types = np.array(percentages_types)
    percentages_types = np.nan_to_num(percentages_types)

    percent_types = percentages_types[..., 0, :]
    percent_type_conds = np.moveaxis(percentages_types[..., 1:, :], 0, 1)

    unscaled_weighted_avgs = np.sum(percent_types * percent_type_conds * factors, axis=-2)
    scaled_weighted_avgs = unscaled_weighted_avgs / np.sum(unscaled_weighted_avgs, axis=0)

    result = np.array(["gut"] * len(dataframe), dtype="object")
    result[scaled_weighted_avgs[1] + scaled_weighted_avgs[2] >= thresholds["tiles_mittel"]] = "mittel"
    result[scaled_weighted_avgs[2] >= thresholds["tiles_schlecht_blocking_gut"]] = "mittel"
    result[scaled_weighted_avgs[2] >= thresholds["tiles_schlecht"]] = "schlecht"
    result[np.isnan(scaled_weighted_avgs[0])] = np.nan

    dataframe["zustand"] = result

    return dataframe


def initialize_dict_with_nan(area_dict):
    """
    Initializes dict for tile with nan.

    :param dict area_dict: Dict containing tile.
    :return: Dict containing tile with added fields, all initialized with nan.
    :rtype: dict
    """
    area_dict["gut"] = np.nan
    area_dict["mittel"] = np.nan
    area_dict["schlecht"] = np.nan
    area_dict["Mark_Antl"] = np.nan
    area_dict["zustand"] = np.nan
    for field_names in type_field_names:
        for field_name in field_names:
            if field_name != "ignore":
                area_dict[field_name] = np.nan

    return area_dict


def get_tile_shape(condition_mask, type_mask, tile_dict, factors, thresholds, height_layer=0, crs="epsg:25832"):
    """
    Creates dataframe of tile to save as shapefile.

    :param np.ndarray condition_mask: Mask containing condition classes.
    :param np.ndarray type_mask: Mask containing type classes.
    :param dict tile_dict: Dict containing tile.
    :param np.ndarray factors: Array with weighting factors for each type-condition combination.
    :param dict thresholds: Dict containing thresholds for classification.
    :param int height_layer: Height-layer the tile belongs to.
    :param str crs: EPGS-code to create shapes in. Must match other coordinates given.
    :return: Dataframe containing tile.
    :rtype: gpd.GeoDataFrame
    """
    area_dict = initialize_dict_with_nan(tile_dict)
    area_dict["ebene"] = str(height_layer)

    num_gut_total = np.count_nonzero(condition_mask == 1)
    num_mittel_total = np.count_nonzero(condition_mask == 2)
    num_schlecht_total = np.count_nonzero(condition_mask == 3)
    num_any_total = np.count_nonzero(condition_mask)

    nums_types = []

    for _type in range(9):
        nums_type = [np.count_nonzero(type_mask == _type + 1)]
        for _cond in range(3):
            nums_type.append(np.count_nonzero((type_mask == _type + 1) * (condition_mask == _cond + 1)))
        nums_types.append(nums_type)

    if num_any_total > 0:
        num_any_total = np.array(num_any_total)

        area_dict["gut"] = 100 * num_gut_total / num_any_total
        area_dict["mittel"] = 100 * num_mittel_total / num_any_total
        area_dict["schlecht"] = 100 * num_schlecht_total / num_any_total
        area_dict["Mark_Antl"] = 100 * num_any_total / (np.product(condition_mask.shape))

        area_dict = assign_percentages_types(nums_types, area_dict)

    area_shape = gpd.GeoDataFrame([area_dict], crs=crs)
    area_shape = classify_tile_dataframe(area_shape, factors, thresholds)

    return area_shape
