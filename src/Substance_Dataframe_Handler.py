# @author: Maryniak, Marius - Fachbereich Elektrotechnik, Westfälische Hochschule Gelsenkirchen

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box

pd.options.mode.chained_assignment = None


def sieve_gdf(gdf,
              sieve_size_r=5,
              sieve_size_onf=5,
              sieve_size_af=10,
              sieve_size_aa=10,
              sieve_size_b=20,
              sieve_size_of=10,
              sieve_size_gvvf=10,
              sieve_size_g=10,
              sieve_size_mt=20,
              sieve_size_ae=10,
              sieve_size_ab=10,
              resolution=.009_765_625):
    """
    | Returns the sieved geodataframe.

    :param gpd.GeoDataFrame gdf: geodataframe
    :param int sieve_size_r: sieve size of damage type r (Risse (Asphalt)) in pixels
    :param int sieve_size_onf: sieve size of damage type onf (Offene Nähte und Fugen (Asphalt)) in pixels
    :param int sieve_size_af: sieve size of damage type af (Aufgelegte Flickstellen (Asphalt)) in pixels
    :param int sieve_size_aa: sieve size of damage type aa (Abplatzungen und Ausbrüche (Asphalt)) in pixels
    :param int sieve_size_b: sieve size of damage type b (Bindemittelanreicherung (Asphalt)) in pixels
    :param int sieve_size_of: sieve size of damage type of (Offene Fugen (Pflaster/ Platten)) in pixels
    :param int sieve_size_gvvf: sieve size of damage type gvvf (Gelockerter Verband/ Verschobenes Fugenbild
        (Pflaster/ Platten)) in pixels
    :param int sieve_size_g: sieve size of damage type g (Gefügeauflösung (Pflaster/ Platten)) in pixels
    :param int sieve_size_mt: sieve size of damage type mt (Materialfremder Teilersatz (Pflaster/ Platten))
        in pixels
    :param int sieve_size_ae: sieve size of damage type ae (Entwässerungseinrichtung (Allgemein)) in pixels
    :param int sieve_size_ab: sieve size of damage type ab (Bordstein (Allgemein)) in pixels
    :param float resolution: resolution in meters per pixel
    :returns: sieved geodataframe
    :rtype: gpd.GeoDataFrame
    """
    damage_types = {'r': sieve_size_r,
                    'onf': sieve_size_onf,
                    'af': sieve_size_af,
                    'aa': sieve_size_aa,
                    'b': sieve_size_b,
                    'of': sieve_size_of,
                    'gvvf': sieve_size_gvvf,
                    'g': sieve_size_g,
                    'mt': sieve_size_mt,
                    'ae': sieve_size_ae,
                    'ab': sieve_size_ab}

    gdf['sieve_size'] = gdf['typ'].map(damage_types)

    mask = gdf.area >= (gdf['sieve_size'] * (resolution ** 2))
    gdf_sieved = gdf.loc[mask]
    gdf_sieved.drop(columns=['sieve_size'], inplace=True)
    gdf_sieved.reset_index(drop=True, inplace=True)
    return gdf_sieved


def clip_gdf(gdf, gdf_substance):
    """
    | Returns the clipped geodataframe.

    :param gpd.GeoDataFrame gdf: geodataframe with polygons to clip the substance values to
    :param gpd.GeoDataFrame gdf_substance: geodataframe of substance values
    :returns: clipped geodataframe
    :rtype: gpd.GeoDataFrame
    """
    gdf_clipped = gpd.clip(gdf=gdf_substance,
                           mask=gdf.geometry,
                           keep_geom_type=True)
    gdf_clipped.sort_index(inplace=True)
    gdf_clipped.reset_index(drop=True, inplace=True)
    return gdf_clipped


def get_gdf_grid(bounding_box, crs='EPSG:25832'):
    """
    | Returns the geodataframe with polygons (1 meter grid) to aggregate the substance values to.

    :param (int, int, int, int) bounding_box: bounding box (x_min, y_min, x_max, y_max)
    :param str crs: coordinate reference system
    :returns: geodataframe with polygons (1 meter grid)
    :rtype: gpd.GeoDataFrame
    """
    x_min, y_min, x_max, y_max = bounding_box

    coordinates_x, coordinates_y = np.meshgrid(np.arange(x_min, x_max), np.arange(y_min, y_max))

    coordinates = np.concatenate((coordinates_x.reshape(-1)[:, np.newaxis],
                                  coordinates_y.reshape(-1)[:, np.newaxis]), axis=-1)

    polygons = [box(x, y, x + 1, y + 1) for x, y in coordinates]
    gdf_grid = gpd.GeoDataFrame(geometry=polygons)
    gdf_grid.set_crs(crs, inplace=True)

    return gdf_grid


def subaggregate_gdf(gdf_grid,
                     gdf_substance,
                     mode,
                     threshold_r=.005,
                     threshold_onf=.005,
                     threshold_af=.02,
                     threshold_aa=.01,
                     threshold_b=.02,
                     threshold_of=.005,
                     threshold_gvvf=.005,
                     threshold_g=.005,
                     threshold_mt=.01,
                     threshold_ae=.005,
                     threshold_ab=.005):
    """
    | Returns the subaggregated geodataframe with appended binary values for each polygon with the following schema:
    | original schema
    | if mode is asphalt:
    | r: damage type r (Risse (Asphalt))
    | onf: damage type onf (Offene Nähte und Fugen (Asphalt))
    | af: damage type af (Aufgelegte Flickstellen (Asphalt))
    | aa: damage type aa (Abplatzungen und Ausbrüche (Asphalt))
    | b: damage type b (Bindemittelanreicherung (Asphalt))
    | ae: damage type ae (Entwässerungseinrichtung (Allgemein))
    | ab: damage type ab (Bordstein (Allgemein))
    | if mode is pflaster_platten:
    | of: damage type of (Offene Fugen (Pflaster/ Platten))
    | gvvf: damage type gvvf (Gelockerter Verband/ Verschobenes Fugenbild (Pflaster/ Platten))
    | g: damage type g (Gefügeauflösung (Pflaster/ Platten))
    | mt: damage type mt (Materialfremder Teilersatz (Pflaster/ Platten))
    | ae: damage type ae (Entwässerungseinrichtung (Allgemein))
    | ab: damage type ab (Bordstein (Allgemein))

    :param gpd.GeoDataFrame gdf_grid: geodataframe with polygons (1 meter grid) to aggregate the substance values to
    :param gpd.GeoDataFrame gdf_substance: geodataframe of substance values
    :param str mode: mode (asphalt or pflaster_platten)
    :param float threshold_r: threshold of damage type r (Risse (Asphalt)) in square meters
    :param float threshold_onf: threshold of damage type onf (Offene Nähte und Fugen (Asphalt)) in square meters
    :param float threshold_af: threshold of damage type af (Aufgelegte Flickstellen (Asphalt)) in square meters
    :param float threshold_aa: threshold of damage type aa (Abplatzungen und Ausbrüche (Asphalt)) in square meters
    :param float threshold_b: threshold of damage type b (Bindemittelanreicherung (Asphalt)) in square meters
    :param float threshold_of: threshold of damage type of (Offene Fugen (Pflaster/ Platten)) in square meters
    :param float threshold_gvvf: threshold of damage type gvvf (Gelockerter Verband/ Verschobenes Fugenbild
        (Pflaster/ Platten)) in square meters
    :param float threshold_g: threshold of damage type g (Gefügeauflösung (Pflaster/ Platten)) in square meters
    :param float threshold_mt: threshold of damage type mt (Materialfremder Teilersatz (Pflaster/ Platten))
        in square meters
    :param float threshold_ae: threshold of damage type ae (Entwässerungseinrichtung (Allgemein)) in square meters
    :param float threshold_ab: threshold of damage type ab (Bordstein (Allgemein)) in square meters
    :returns: subaggregated geodataframe
    :rtype: gpd.GeoDataFrame
    """
    damage_types = {'r': threshold_r,
                    'onf': threshold_onf,
                    'af': threshold_af,
                    'aa': threshold_aa,
                    'b': threshold_b,
                    'of': threshold_of,
                    'gvvf': threshold_gvvf,
                    'g': threshold_g,
                    'mt': threshold_mt,
                    'ae': threshold_ae,
                    'ab': threshold_ab}

    if mode == 'asphalt':
        damage_types = {key: value
                        for key, value in damage_types.items()
                        if key in ['r', 'onf', 'af', 'aa', 'b', 'ae', 'ab']}
    elif mode == 'pflaster_platten':
        damage_types = {key: value
                        for key, value in damage_types.items()
                        if key in ['of', 'gvvf', 'g', 'mt', 'ae', 'ab']}
    else:
        raise ValueError('mode must be either asphalt or pflaster_platten!')

    gdf_grid['aggregation_id'] = gdf_grid.index

    gdf_intersection = gpd.overlay(df1=gdf_grid,
                                   df2=gdf_substance,
                                   how='intersection',
                                   keep_geom_type=True)
    gdf_intersection['damage_area'] = gdf_intersection.geometry.area

    gdf_aggregated_area = \
        gdf_intersection.groupby(['aggregation_id', 'typ'])['damage_area'].sum().reset_index(drop=False)

    gdf_aggregated_area_pivoted = gdf_aggregated_area.pivot_table(index='aggregation_id',
                                                                  columns='typ',
                                                                  values='damage_area').reset_index(drop=False)

    gdf_grid = gdf_grid.merge(gdf_aggregated_area_pivoted,
                              on='aggregation_id',
                              how='left')

    for damage_type, damage_type_threshold in damage_types.items():
        if damage_type in list(gdf_substance['typ'].unique()):
            gdf_grid[damage_type] = (gdf_grid[damage_type].fillna(0) >= damage_type_threshold)
        else:
            gdf_grid[damage_type] = False

    gdf_grid = gdf_grid[(gdf_grid[[*damage_types]]).any(axis=1)]
    gdf_grid = gdf_grid[['geometry'] + [*damage_types]]
    gdf_grid.reset_index(drop=True, inplace=True)
    return gdf_grid


def aggregate_gdf(gdf_aggregation_areas,
                  gdf_subaggregated,
                  mode):
    """
    | Returns the aggregated geodataframe with appended statistical values for each polygon with the following schema:
    | original schema
    | if mode is asphalt:
    | damage_area_r: damage area of damage type r (Risse) in percent
    | damage_area_af: damage area of damage type af (Aufgelegte Flickstellen) in percent
    | damage_area_o: damage area of damage type onf, aa, b (Offene Nähte und Fugen, Abplatzungen und Ausbrüche,
        Bindemittelanreicherung) in percent
    | if mode is pflaster_platten:
    | damage_area_o: damage area of damage type of, gvvf, g, mt (Offene Fugen, Gelockerter Verband/
        Verschobenes Fugenbild, Gefügeauflösung, Materialfremder Teilersatz) in percent

    :param gpd.GeoDataFrame gdf_aggregation_areas: geodataframe with polygons to aggregate
        the subaggregated substance values to
    :param gpd.GeoDataFrame gdf_subaggregated: geodataframe of subaggregated substance values
    :param str mode: mode (asphalt or pflaster_platten)
    :returns: aggregated geodataframe
    :rtype: gpd.GeoDataFrame
    """
    if mode == 'asphalt':
        damage_types_grouped = {'r': ['r'],
                                'af': ['af'],
                                'o': ['onf', 'aa', 'b']}
    elif mode == 'pflaster_platten':
        damage_types_grouped = {'o': ['of', 'gvvf', 'g', 'mt']}
    else:
        raise ValueError('mode must be either asphalt or pflaster_platten!')

    gdf_aggregation_areas['aggregation_id'] = gdf_aggregation_areas.index

    gdf_intersection = gpd.overlay(df1=gdf_aggregation_areas,
                                   df2=gdf_subaggregated,
                                   how='intersection',
                                   keep_geom_type=True)

    for damage_type, damage_type_group in damage_types_grouped.items():
        gdf_intersection[damage_type] = gdf_intersection[damage_type_group].max(axis=1)  # elementwise or

    gdf_aggregated_area = \
        gdf_intersection.groupby(['aggregation_id'])[[*damage_types_grouped]].sum().reset_index(drop=False)

    gdf_aggregated = gdf_aggregation_areas.merge(gdf_aggregated_area,
                                                 on='aggregation_id',
                                                 how='left')
    gdf_aggregated['area'] = gdf_aggregated.geometry.area

    damage_area_columns = [f'damage_area_{damage_type}' for damage_type in [*damage_types_grouped]]
    gdf_aggregated[[*damage_area_columns]] = gdf_aggregated[[*damage_types_grouped]].div(gdf_aggregated['area'],
                                                                                         axis=0)
    gdf_aggregated[[*damage_area_columns]].clip(upper=1,
                                                inplace=True)
    gdf_aggregated[[*damage_area_columns]] *= 100

    for damage_area_column in damage_area_columns:
        gdf_aggregated[damage_area_column] = gdf_aggregated[damage_area_column].fillna(0)

    gdf_aggregated.drop(columns=['aggregation_id', 'area'] + [*damage_types_grouped], inplace=True)
    return gdf_aggregated


def get_ap9_compliant_gdf(gdf,
                          mode,
                          precision=2,
                          drop_original_schema=False):
    """
    | Returns the AP9 compliant aggregated geodataframe with appended statistical values for each polygon with
        the following schema (according to 'Arbeitspapier 9 K zur Systematik der Straßenerhaltung'):
    | original schema
    | if mode is asphalt:
    | RISS: Anteil der durch Risse betroffenen Fläche [%]
    | AFLI: Anteil der durch aufgelegte Flickstellen betroffenen Fläche [%]
    | OFS: Anteil der durch sonstige Oberflächenschäden betroffenen Fläche [%]
    | if mode is pflaster_platten:
    | OFS: Anteil der durch Oberflächenschäden betroffenen Fläche [%]

    :param gpd.GeoDataFrame gdf: aggregated geodataframe
    :param str mode: mode (asphalt or pflaster_platten)
    :param int precision: precision of the float values
    :param bool drop_original_schema: if True, the original schema and their attributes are dropped
    :returns: aggregated geodataframe
    :rtype: gpd.GeoDataFrame
    """
    if mode == 'asphalt':
        ap9_compliant_schema = {'damage_area_r': 'RISS',
                                'damage_area_af': 'AFLI',
                                'damage_area_o': 'OFS'}
    elif mode == 'pflaster_platten':
        ap9_compliant_schema = {'damage_area_o': 'OFS'}
    else:
        raise ValueError('mode must be either asphalt or pflaster_platten!')

    if drop_original_schema:
        gdf.drop(columns=gdf.columns.difference(['geometry'] + [*ap9_compliant_schema]), inplace=True)

    for column in gdf.select_dtypes(include=['float64']).columns:
        gdf[column] = gdf[column].apply(lambda x: round(x, precision) if not pd.isnull(x) else x)

    gdf.rename(columns=ap9_compliant_schema, inplace=True)
    return gdf


def process_gdf_substance(gdf_substance,
                          gdf_aggregation_areas,
                          crs,
                          mode):
    """
    | Returns the AP9 compliant aggregated geodataframe with appended statistical values for each polygon with
        the following schema (according to 'Arbeitspapier 9 K zur Systematik der Straßenerhaltung'):
    | original schema
    | if mode is asphalt:
    | RISS: Anteil der durch Risse betroffenen Fläche [%]
    | AFLI: Anteil der durch aufgelegte Flickstellen betroffenen Fläche [%]
    | OFS: Anteil der durch sonstige Oberflächenschäden betroffenen Fläche [%]
    | if mode is pflaster_platten:
    | OFS: Anteil der durch Oberflächenschäden betroffenen Fläche [%]

    :param gpd.GeoDataFrame gdf_substance: geodataframe of substance values
    :param gpd.GeoDataFrame gdf_aggregation_areas: geodataframe with polygons to aggregate
        the subaggregated substance values to
    :param str crs: coordinate reference system
    :param str mode: mode (asphalt or pflaster_platten)
    :returns: aggregated geodataframe
    :rtype: gpd.GeoDataFrame
    """
    if gdf_aggregation_areas.empty:
        return gpd.GeoDataFrame(geometry=[],
                                crs=crs)

    if mode not in ['asphalt', 'pflaster_platten']:
        raise ValueError('mode must be either asphalt or pflaster_platten!')

    gdf_substance = sieve_gdf(gdf=gdf_substance)
    gdf_substance = clip_gdf(gdf=gdf_aggregation_areas, gdf_substance=gdf_substance)

    bounding_box = gdf_aggregation_areas.total_bounds
    bounding_box = tuple(np.array([np.floor(bounding_box[0]),
                                   np.floor(bounding_box[1]),
                                   np.ceil(bounding_box[2]),
                                   np.ceil(bounding_box[3])],
                                  dtype=int))
    # noinspection PyTypeChecker
    gdf_grid = get_gdf_grid(bounding_box=bounding_box,
                            crs=crs)

    gdf_grid = gpd.sjoin(gdf_grid,
                         gdf_aggregation_areas,
                         how='inner',
                         predicate='intersects')
    gdf_grid = gdf_grid[['geometry']].reset_index(drop=True)

    gdf_subaggregated = subaggregate_gdf(gdf_grid=gdf_grid,
                                         gdf_substance=gdf_substance,
                                         mode=mode)

    gdf_aggregated = aggregate_gdf(gdf_aggregation_areas=gdf_aggregation_areas,
                                   gdf_subaggregated=gdf_subaggregated,
                                   mode=mode)

    gdf_aggregated = get_ap9_compliant_gdf(gdf=gdf_aggregated,
                                           mode=mode)

    return gdf_aggregated
