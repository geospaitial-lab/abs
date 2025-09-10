# @author: Maryniak, Marius - Fachbereich Elektrotechnik, Westfälische Hochschule Gelsenkirchen

import geopandas as gpd
import numpy as np
import pandas as pd

pd.options.mode.chained_assignment = None


def merge_gdfs(gdf_planeness, gdf_substance):
    """
    | Returns the AP9 compliant merged geodataframe with statistical values for each polygon with
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
    | RISS: Anteil der durch Risse betroffenen Fläche [%]
    | AFLI: Anteil der durch aufgelegte Flickstellen betroffenen Fläche [%]
    | OFS: Anteil der durch sonstige Oberflächenschäden betroffenen Fläche [%]

    :param gpd.GeoDataFrame gdf_planeness: geodataframe of aggregated planeness values
    :param gpd.GeoDataFrame gdf_substance: geodataframe of aggregated substance values
    :returns: merged geodataframe
    :rtype: gpd.GeoDataFrame
    """
    gdf_merged = gdf_planeness.merge(gdf_substance)
    gdf_merged.insert(0, 'geometry', gdf_merged.pop('geometry'))
    return gdf_merged


def mask_gdf(gdf,
             column,
             value):
    """
    | Returns the masked geodataframe.
    |
    | Based on:
    | https://stackoverflow.com/a/46165056

    :param gpd.GeoDataFrame gdf: geodataframe
    :param str column: name of the column
    :param str or int or float value: value
    :returns: masked geodataframe
    :rtype: gpd.GeoDataFrame
    """
    mask = gdf[column].values == value

    return gdf[mask]


def normalize(values, normalization_factors):
    """
    | Returns the normalized values.

    :param np.ndarray[np.float64] values: values
    :param (float, float, float) normalization_factors: normalization factors (1.5-Wert, Warnwert (3.5),
        Schwellenwert (4.5))
    :returns: normalized values
    :rtype: np.ndarray[np.float64]
    """
    fp_1, fp_2, fp_3 = normalization_factors
    os = 1.5 * fp_3 - .5 * fp_2

    conditions = [values < fp_1,
                  (values >= fp_1) & (values <= fp_2),
                  (values >= fp_2) & (values <= fp_3),
                  (values >= fp_3) & (values <= os),
                  values > os]

    functions = [lambda x: 1 + .5 * (x / fp_1),
                 lambda x: 1.5 + 2 * (x - fp_1) / (fp_2 - fp_1),
                 lambda x: 3.5 + (x - fp_2) / (fp_3 - fp_2),
                 lambda x: 4.5 + .5 * (x - fp_3) / (os - fp_3),
                 lambda x: 5,
                 lambda x: np.nan]

    values = np.piecewise(values,
                          condlist=conditions,
                          funclist=functions)
    return values


def get_values_twgeb_asphalt(gdf, weighting_factors):
    """
    | Returns the values of TWGEB (Gebrauchswert) (according to 'Arbeitspapier 9 K zur Systematik
        der Straßenerhaltung').

    :param gpd.GeoDataFrame gdf: geodataframe
    :param dict weighting_factors: weighting factors
    :returns: twgeb values
    :rtype: np.ndarray[np.float64]
    """
    values_zwsm4l_zwmspt_max = np.maximum(gdf['ZWSM4L'], gdf['ZWMSPT'])

    values_twgeb = np.where(values_zwsm4l_zwmspt_max > 3.5,
                            values_zwsm4l_zwmspt_max,
                            gdf['ZWSM4L'] * weighting_factors['ASPHALT']['TWGEB']['ZWSM4L']
                            + gdf['ZWMSPT'] * weighting_factors['ASPHALT']['TWGEB']['ZWMSPT']
                            + gdf['ZWMSPH'] * weighting_factors['ASPHALT']['TWGEB']['ZWMSPH'])
    return values_twgeb


def get_values_twsub_asphalt(gdf, weighting_factors):
    """
    | Returns the values of TWSUB (Substanzwert) (according to 'Arbeitspapier 9 K zur Systematik
        der Straßenerhaltung').

    :param gpd.GeoDataFrame gdf: geodataframe
    :param dict weighting_factors: weighting factors
    :returns: twsub values
    :rtype: np.ndarray[np.float64]
    """
    values_zwriss_zwofs_max = np.maximum(gdf['ZWRISS'], gdf['ZWOFS'])

    values_twsub = np.where(values_zwriss_zwofs_max > 3.5,
                            values_zwriss_zwofs_max,
                            gdf['ZWRISS'] * weighting_factors['ASPHALT']['TWSUB']['ZWRISS']
                            + gdf['ZWAFLI'] * weighting_factors['ASPHALT']['TWSUB']['ZWAFLI']
                            + gdf['ZWOFS'] * weighting_factors['ASPHALT']['TWSUB']['ZWOFS'])
    return values_twsub


def get_values_twgeb_pflaster_platten(gdf, weighting_factors):
    """
    | Returns the values of TWGEB (Gebrauchswert) (according to 'Arbeitspapier 9 K zur Systematik
        der Straßenerhaltung').

    :param gpd.GeoDataFrame gdf: geodataframe
    :param dict weighting_factors: weighting factors
    :returns: twgeb values
    :rtype: np.ndarray[np.float64]
    """
    values_zwsm4l_zwmspt_max = np.maximum(gdf['ZWSM4L'], gdf['ZWMSPT'])

    values_twgeb = np.where(values_zwsm4l_zwmspt_max > 3.5,
                            values_zwsm4l_zwmspt_max,
                            gdf['ZWSM4L'] * weighting_factors['PFLASTER_PLATTEN']['TWGEB']['ZWSM4L']
                            + gdf['ZWMSPT'] * weighting_factors['PFLASTER_PLATTEN']['TWGEB']['ZWMSPT']
                            + gdf['ZWMSPH'] * weighting_factors['PFLASTER_PLATTEN']['TWGEB']['ZWMSPH'])
    return values_twgeb


def quantize(values):
    """
    | Returns the quantized values with bin edges of 1.5, 3.5 and 4.5 (according to 'Arbeitspapier 9 K
        zur Systematik der Straßenerhaltung').

    :param np.ndarray[np.float64] values: values
    :returns: quantized values
    :rtype: np.ndarray[np.float64]
    """
    values_quantized = np.where(np.isnan(values),
                                0,
                                (np.digitize(values,
                                             bins=[1.5, 3.5, 4.5],
                                             right=True) + 1))

    return values_quantized


def concatenate_gdfs(gdfs):
    """
    | Returns the concatenated geodataframe.
    | The original order of the indices is preserved.

    :param list[gpd.GeoDataFrame] gdfs: geodataframes
    :returns: concatenated geodataframe
    :rtype: gpd.GeoDataFrame
    """
    gdf_concatenated = pd.concat(gdfs)
    gdf_concatenated.sort_index(inplace=True)
    return gdf_concatenated


def evaluate_gdf_asphalt(gdf,
                         normalization_factors,
                         weighting_factors,
                         mode):
    """
    | Returns the AP9 compliant evaluated geodataframe with appended statistical values for each polygon with
        the following schema (according to 'Arbeitspapier 9 K zur Systematik der Straßenerhaltung'):
    | original schema
    | ZWSM4L_M: Zustandswert (Maximum der Stichmaße in Lattenmitte unter der 4m Latte)
    | ZWSM4L_A: Zustandswert (Mittelwert der Stichmaße in Lattenmitte unter der 4m Latte)
    | ZWSM4L: Zustandswert (Maximum von ZWSM4L_M und ZWSM4L_A)
    | ZWMSPT: Zustandswert (Maximum der Mittelwerte der linken und rechten Spurrinnentiefen)
    | ZWMSPH: Zustandswert (Maximum der Mittelwerte der linken und rechten fiktiven Wassertiefen)
    | ZWRISS: Zustandswert (Anteil der durch Risse betroffenen Fläche)
    | ZWAFLI: Zustandswert (Anteil der durch aufgelegte Flickstellen betroffenen Fläche)
    | ZWOFS: Zustandswert (Anteil der durch sonstige Oberflächenschäden betroffenen Fläche)
    | TWGEB: Gebrauchswert
    | TWSUB: Substanzwert
    | GW: Gesamtwert (Maximum von TWGEB und TWSUB)
    | ZKSM4L_M: Zustandsklasse (Maximum der Stichmaße in Lattenmitte unter der 4m Latte)
    | ZKSM4L_A: Zustandsklasse (Mittelwert der Stichmaße in Lattenmitte unter der 4m Latte)
    | ZKSM4L: Zustandsklasse (Maximum von ZWSM4L_M und ZWSM4L_A)
    | ZKMSPT: Zustandsklasse (Maximum der Mittelwerte der linken und rechten Spurrinnentiefen)
    | ZKMSPH: Zustandsklasse (Maximum der Mittelwerte der linken und rechten fiktiven Wassertiefen)
    | ZKRISS: Zustandsklasse (Anteil der durch Risse betroffenen Fläche)
    | ZKAFLI: Zustandsklasse (Anteil der durch aufgelegte Flickstellen betroffenen Fläche)
    | ZKOFS: Zustandsklasse (Anteil der durch sonstige Oberflächenschäden betroffenen Fläche)
    | TKGEB: Gebrauchsklasse
    | TKSUB: Substanzklasse
    | GK: Gesamtklasse (Maximum von TWGEB und TWSUB)

    :param gpd.GeoDataFrame gdf: geodataframe
    :param dict normalization_factors: normalization factors (1.5-Wert, Warnwert (3.5),
        Schwellenwert (4.5))
    :param dict weighting_factors: weighting factors
    :param str mode: mode (asphalt or pflaster_platten)
    :returns: evaluated geodataframe
    :rtype: gpd.GeoDataFrame
    """
    if mode not in ['A', 'B', 'N']:
        raise ValueError('mode must be either A, B or N!')

    if mode in ['A', 'B']:
        values_zwsm4l_m = normalize(np.array(gdf['SM4L_M']), normalization_factors['ASPHALT']['SM4L_M'][mode])
        values_zwsm4l_a = normalize(np.array(gdf['SM4L_A']), normalization_factors['ASPHALT']['SM4L_A'][mode])
        values_zwsm4l = np.maximum(values_zwsm4l_m, values_zwsm4l_a)
        values_zwmspt = normalize(np.array(gdf['MSPT']), normalization_factors['ASPHALT']['MSPT'][mode])
        values_zwmsph = normalize(np.array(gdf['MSPH']), normalization_factors['ASPHALT']['MSPH'][mode])

        gdf['ZWSM4L_M'] = values_zwsm4l_m
        gdf['ZWSM4L_A'] = values_zwsm4l_a
        gdf['ZWSM4L'] = values_zwsm4l
        gdf['ZWMSPT'] = values_zwmspt
        gdf['ZWMSPH'] = values_zwmsph
    else:
        gdf['ZWSM4L_M'] = np.nan
        gdf['ZWSM4L_A'] = np.nan
        gdf['ZWSM4L'] = np.nan
        gdf['ZWMSPT'] = np.nan
        gdf['ZWMSPH'] = np.nan

    values_zwriss = normalize(np.array(gdf['RISS']), normalization_factors['ASPHALT']['RISS'][mode])
    values_zwafli = normalize(np.array(gdf['AFLI']), normalization_factors['ASPHALT']['AFLI'][mode])
    values_zwofs = normalize(np.array(gdf['OFS']), normalization_factors['ASPHALT']['OFS'][mode])

    gdf['ZWRISS'] = values_zwriss
    gdf['ZWAFLI'] = values_zwafli
    gdf['ZWOFS'] = values_zwofs

    for column in ['ZWSM4L_M', 'ZWSM4L_A', 'ZWSM4L', 'ZWMSPT', 'ZWMSPH', 'ZWRISS', 'ZWAFLI', 'ZWOFS']:
        gdf[column] = gdf[column].apply(lambda x: round(x, 2) if not pd.isnull(x) else x)

    if mode in ['A', 'B']:
        gdf['TWGEB'] = get_values_twgeb_asphalt(gdf, weighting_factors)
    else:
        gdf['TWGEB'] = np.nan

    gdf['TWSUB'] = get_values_twsub_asphalt(gdf, weighting_factors)

    gdf['GW'] = np.maximum(gdf['TWGEB'].fillna(gdf['TWSUB']), gdf['TWSUB'])

    for column in ['TWGEB', 'TWSUB', 'GW']:
        gdf[column] = gdf[column].apply(lambda x: round(x, 2) if not pd.isnull(x) else x)

    values = np.array(gdf[['ZWSM4L_M', 'ZWSM4L_A', 'ZWSM4L', 'ZWMSPT', 'ZWMSPH',
                           'ZWRISS', 'ZWAFLI', 'ZWOFS',
                           'TWGEB', 'TWSUB', 'GW']])

    gdf[['ZKSM4L_M', 'ZKSM4L_A', 'ZKSM4L', 'ZKMSPT', 'ZKMSPH',
         'ZKRISS', 'ZKAFLI', 'ZKOFS',
         'TKGEB', 'TKSUB', 'GK']] = quantize(values)

    gdf[['ZKSM4L_M', 'ZKSM4L_A', 'ZKSM4L', 'ZKMSPT', 'ZKMSPH', 'TKGEB']] = (
        gdf)[['ZKSM4L_M', 'ZKSM4L_A', 'ZKSM4L', 'ZKMSPT', 'ZKMSPH', 'TKGEB']].replace(0, np.nan)

    return gdf


def evaluate_gdf_pflaster_platten(gdf,
                                  normalization_factors,
                                  weighting_factors,
                                  mode):
    """
    | Returns the AP9 compliant evaluated geodataframe with appended statistical values for each polygon with
        the following schema (according to 'Arbeitspapier 9 K zur Systematik der Straßenerhaltung'):
    | original schema
    | ZWSM4L_M: Zustandswert (Maximum der Stichmaße in Lattenmitte unter der 4m Latte)
    | ZWSM4L_A: Zustandswert (Mittelwert der Stichmaße in Lattenmitte unter der 4m Latte)
    | ZWSM4L: Zustandswert (Maximum von ZWSM4L_M und ZWSM4L_A)
    | ZWMSPT: Zustandswert (Maximum der Mittelwerte der linken und rechten Spurrinnentiefen)
    | ZWMSPH: Zustandswert (Maximum der Mittelwerte der linken und rechten fiktiven Wassertiefen)
    | ZWRISS: Zustandswert (Anteil der durch Risse betroffenen Fläche)
    | ZWAFLI: Zustandswert (Anteil der durch aufgelegte Flickstellen betroffenen Fläche)
    | ZWOFS: Zustandswert (Anteil der durch sonstige Oberflächenschäden betroffenen Fläche)
    | TWGEB: Gebrauchswert
    | TWSUB: Substanzwert
    | GW: Gesamtwert (Maximum von TWGEB und TWSUB)
    | ZKSM4L_M: Zustandsklasse (Maximum der Stichmaße in Lattenmitte unter der 4m Latte)
    | ZKSM4L_A: Zustandsklasse (Mittelwert der Stichmaße in Lattenmitte unter der 4m Latte)
    | ZKSM4L: Zustandsklasse (Maximum von ZWSM4L_M und ZWSM4L_A)
    | ZKMSPT: Zustandsklasse (Maximum der Mittelwerte der linken und rechten Spurrinnentiefen)
    | ZKMSPH: Zustandsklasse (Maximum der Mittelwerte der linken und rechten fiktiven Wassertiefen)
    | ZKRISS: Zustandsklasse (Anteil der durch Risse betroffenen Fläche)
    | ZKAFLI: Zustandsklasse (Anteil der durch aufgelegte Flickstellen betroffenen Fläche)
    | ZKOFS: Zustandsklasse (Anteil der durch sonstige Oberflächenschäden betroffenen Fläche)
    | TKGEB: Gebrauchsklasse
    | TKSUB: Substanzklasse
    | GK: Gesamtklasse (Maximum von TWGEB und TWSUB)

    :param gpd.GeoDataFrame gdf: geodataframe
    :param dict normalization_factors: normalization factors (1.5-Wert, Warnwert (3.5),
        Schwellenwert (4.5))
    :param dict weighting_factors: weighting factors
    :param str mode: mode (asphalt or pflaster_platten)
    :returns: evaluated geodataframe
    :rtype: gpd.GeoDataFrame
    """
    if mode not in ['A', 'B', 'N']:
        raise ValueError('mode must be either A, B or N!')

    if mode in ['A', 'B']:
        values_zwsm4l_m = normalize(np.array(gdf['SM4L_M']), normalization_factors['PFLASTER_PLATTEN']['SM4L_M'][mode])
        values_zwsm4l_a = normalize(np.array(gdf['SM4L_A']), normalization_factors['PFLASTER_PLATTEN']['SM4L_A'][mode])
        values_zwsm4l = np.maximum(values_zwsm4l_m, values_zwsm4l_a)
        values_zwmspt = normalize(np.array(gdf['MSPT']), normalization_factors['PFLASTER_PLATTEN']['MSPT'][mode])
        values_zwmsph = normalize(np.array(gdf['MSPH']), normalization_factors['PFLASTER_PLATTEN']['MSPH'][mode])

        gdf['ZWSM4L_M'] = values_zwsm4l_m
        gdf['ZWSM4L_A'] = values_zwsm4l_a
        gdf['ZWSM4L'] = values_zwsm4l
        gdf['ZWMSPT'] = values_zwmspt
        gdf['ZWMSPH'] = values_zwmsph
    else:
        gdf['ZWSM4L_M'] = np.nan
        gdf['ZWSM4L_A'] = np.nan
        gdf['ZWSM4L'] = np.nan
        gdf['ZWMSPT'] = np.nan
        gdf['ZWMSPH'] = np.nan

    values_zwofs = normalize(np.array(gdf['OFS']), normalization_factors['PFLASTER_PLATTEN']['OFS'][mode])

    gdf['ZWOFS'] = values_zwofs

    for column in ['ZWSM4L_M', 'ZWSM4L_A', 'ZWSM4L', 'ZWMSPT', 'ZWMSPH', 'ZWOFS']:
        gdf[column] = gdf[column].apply(lambda x: round(x, 2) if not pd.isnull(x) else x)

    if mode in ['A', 'B']:
        gdf['TWGEB'] = get_values_twgeb_pflaster_platten(gdf, weighting_factors)
    else:
        gdf['TWGEB'] = np.nan

    gdf['TWSUB'] = gdf['ZWOFS']

    gdf['GW'] = np.maximum(gdf['TWGEB'].fillna(gdf['TWSUB']), gdf['TWSUB'])

    for column in ['TWGEB', 'TWSUB', 'GW']:
        gdf[column] = gdf[column].apply(lambda x: round(x, 2) if not pd.isnull(x) else x)

    values = np.array(gdf[['ZWSM4L_M', 'ZWSM4L_A', 'ZWSM4L', 'ZWMSPT', 'ZWMSPH',
                           'ZWOFS',
                           'TWGEB', 'TWSUB', 'GW']])

    gdf[['ZKSM4L_M', 'ZKSM4L_A', 'ZKSM4L', 'ZKMSPT', 'ZKMSPH',
         'ZKOFS',
         'TKGEB', 'TKSUB', 'GK']] = quantize(values)

    gdf[['ZKSM4L_M', 'ZKSM4L_A', 'ZKSM4L', 'ZKMSPT', 'ZKMSPH', 'TKGEB']] = (
        gdf)[['ZKSM4L_M', 'ZKSM4L_A', 'ZKSM4L', 'ZKMSPT', 'ZKMSPH', 'TKGEB']].replace(0, np.nan)

    return gdf


def evaluate(gdf_planeness,
             gdf_substance_asphalt,
             gdf_substance_pflaster_platten,
             normalization_factors,
             weighting_factors):
    """
    | Returns the AP9 compliant evaluated geodataframe with appended statistical values for each polygon with
        the following schema (according to 'Arbeitspapier 9 K zur Systematik der Straßenerhaltung'):
    | original schema
    | ZWSM4L_M: Zustandswert (Maximum der Stichmaße in Lattenmitte unter der 4m Latte)
    | ZWSM4L_A: Zustandswert (Mittelwert der Stichmaße in Lattenmitte unter der 4m Latte)
    | ZWSM4L: Zustandswert (Maximum von ZWSM4L_M und ZWSM4L_A)
    | ZWMSPT: Zustandswert (Maximum der Mittelwerte der linken und rechten Spurrinnentiefen)
    | ZWMSPH: Zustandswert (Maximum der Mittelwerte der linken und rechten fiktiven Wassertiefen)
    | ZWRISS: Zustandswert (Anteil der durch Risse betroffenen Fläche)
    | ZWAFLI: Zustandswert (Anteil der durch aufgelegte Flickstellen betroffenen Fläche)
    | ZWOFS: Zustandswert (Anteil der durch sonstige Oberflächenschäden betroffenen Fläche)
    | TWGEB: Gebrauchswert
    | TWSUB: Substanzwert
    | GW: Gesamtwert (Maximum von TWGEB und TWSUB)
    | ZKSM4L_M: Zustandsklasse (Maximum der Stichmaße in Lattenmitte unter der 4m Latte)
    | ZKSM4L_A: Zustandsklasse (Mittelwert der Stichmaße in Lattenmitte unter der 4m Latte)
    | ZKSM4L: Zustandsklasse (Maximum von ZWSM4L_M und ZWSM4L_A)
    | ZKMSPT: Zustandsklasse (Maximum der Mittelwerte der linken und rechten Spurrinnentiefen)
    | ZKMSPH: Zustandsklasse (Maximum der Mittelwerte der linken und rechten fiktiven Wassertiefen)
    | ZKRISS: Zustandsklasse (Anteil der durch Risse betroffenen Fläche)
    | ZKAFLI: Zustandsklasse (Anteil der durch aufgelegte Flickstellen betroffenen Fläche)
    | ZKOFS: Zustandsklasse (Anteil der durch sonstige Oberflächenschäden betroffenen Fläche)
    | TKGEB: Gebrauchsklasse
    | TKSUB: Substanzklasse
    | GK: Gesamtklasse (Maximum von TWGEB und TWSUB)

    :param gpd.GeoDataFrame gdf_planeness: geodataframe of aggregated planeness values
    :param gpd.GeoDataFrame gdf_substance_asphalt: geodataframe of aggregated substance values (Asphalt)
    :param gpd.GeoDataFrame gdf_substance_pflaster_platten: geodataframe of aggregated substance values
        (Pflaster/ Platten)
    :param dict normalization_factors: normalization factors (1.5-Wert, Warnwert (3.5),
        Schwellenwert (4.5))
    :param dict weighting_factors: weighting factors
    :returns: evaluated geodataframe
    :rtype: gpd.GeoDataFrame
    """
    gdfs_evaluated = []

    if not gdf_substance_asphalt.empty:
        gdf_merged_asphalt = merge_gdfs(gdf_planeness=gdf_planeness,
                                        gdf_substance=gdf_substance_asphalt)

        gdf_merged_asphalt_a = mask_gdf(gdf_merged_asphalt,
                                        column='FK',
                                        value='A')
        gdf_merged_asphalt_b = mask_gdf(gdf_merged_asphalt,
                                        column='FK',
                                        value='B')
        gdf_merged_asphalt_n = mask_gdf(gdf_merged_asphalt,
                                        column='FK',
                                        value='N')

        gdf_evaluated_asphalt_a = evaluate_gdf_asphalt(gdf_merged_asphalt_a,
                                                       normalization_factors=normalization_factors,
                                                       weighting_factors=weighting_factors,
                                                       mode='A')
        gdf_evaluated_asphalt_b = evaluate_gdf_asphalt(gdf_merged_asphalt_b,
                                                       normalization_factors=normalization_factors,
                                                       weighting_factors=weighting_factors,
                                                       mode='B')
        gdf_evaluated_asphalt_n = evaluate_gdf_asphalt(gdf_merged_asphalt_n,
                                                       normalization_factors=normalization_factors,
                                                       weighting_factors=weighting_factors,
                                                       mode='N')

        gdfs_evaluated.extend([gdf_evaluated_asphalt_a,
                               gdf_evaluated_asphalt_b,
                               gdf_evaluated_asphalt_n])

    if not gdf_substance_pflaster_platten.empty:
        gdf_merged_pflaster_platten = merge_gdfs(gdf_planeness=gdf_planeness,
                                                 gdf_substance=gdf_substance_pflaster_platten)

        gdf_merged_pflaster_platten_a = mask_gdf(gdf_merged_pflaster_platten,
                                                 column='FK',
                                                 value='A')
        gdf_merged_pflaster_platten_b = mask_gdf(gdf_merged_pflaster_platten,
                                                 column='FK',
                                                 value='B')
        gdf_merged_pflaster_platten_n = mask_gdf(gdf_merged_pflaster_platten,
                                                 column='FK',
                                                 value='N')

        gdf_evaluated_pflaster_platten_a = evaluate_gdf_pflaster_platten(gdf_merged_pflaster_platten_a,
                                                                         normalization_factors=normalization_factors,
                                                                         weighting_factors=weighting_factors,
                                                                         mode='A')
        gdf_evaluated_pflaster_platten_b = evaluate_gdf_pflaster_platten(gdf_merged_pflaster_platten_b,
                                                                         normalization_factors=normalization_factors,
                                                                         weighting_factors=weighting_factors,
                                                                         mode='B')
        gdf_evaluated_pflaster_platten_n = evaluate_gdf_pflaster_platten(gdf_merged_pflaster_platten_n,
                                                                         normalization_factors=normalization_factors,
                                                                         weighting_factors=weighting_factors,
                                                                         mode='N')

        gdfs_evaluated.extend([gdf_evaluated_pflaster_platten_a,
                               gdf_evaluated_pflaster_platten_b,
                               gdf_evaluated_pflaster_platten_n])

    gdf_evaluated = concatenate_gdfs(gdfs_evaluated)

    return gdf_evaluated


def reevaluate(gdf,
               normalization_factors,
               weighting_factors):
    gdfs_evaluated = []

    gdf_asphalt = gdf[gdf['BW'] == 'A']

    if not gdf_asphalt.empty:
        gdf_asphalt_a = mask_gdf(gdf_asphalt,
                                 column='FK',
                                 value='A')
        gdf_asphalt_b = mask_gdf(gdf_asphalt,
                                 column='FK',
                                 value='B')
        gdf_asphalt_n = mask_gdf(gdf_asphalt,
                                 column='FK',
                                 value='N')

        gdf_evaluated_asphalt_a = evaluate_gdf_asphalt(gdf_asphalt_a,
                                                       normalization_factors=normalization_factors,
                                                       weighting_factors=weighting_factors,
                                                       mode='A')
        gdf_evaluated_asphalt_b = evaluate_gdf_asphalt(gdf_asphalt_b,
                                                       normalization_factors=normalization_factors,
                                                       weighting_factors=weighting_factors,
                                                       mode='B')
        gdf_evaluated_asphalt_n = evaluate_gdf_asphalt(gdf_asphalt_n,
                                                       normalization_factors=normalization_factors,
                                                       weighting_factors=weighting_factors,
                                                       mode='N')

        gdfs_evaluated.extend([gdf_evaluated_asphalt_a,
                               gdf_evaluated_asphalt_b,
                               gdf_evaluated_asphalt_n])

    gdf_pflaster_platten = gdf[gdf['BW'] == 'P']

    if not gdf_pflaster_platten.empty:
        gdf_pflaster_platten_a = mask_gdf(gdf_pflaster_platten,
                                          column='FK',
                                          value='A')
        gdf_pflaster_platten_b = mask_gdf(gdf_pflaster_platten,
                                          column='FK',
                                          value='B')
        gdf_pflaster_platten_n = mask_gdf(gdf_pflaster_platten,
                                          column='FK',
                                          value='N')

        gdf_evaluated_pflaster_platten_a = evaluate_gdf_pflaster_platten(gdf_pflaster_platten_a,
                                                                         normalization_factors=normalization_factors,
                                                                         weighting_factors=weighting_factors,
                                                                         mode='A')
        gdf_evaluated_pflaster_platten_b = evaluate_gdf_pflaster_platten(gdf_pflaster_platten_b,
                                                                         normalization_factors=normalization_factors,
                                                                         weighting_factors=weighting_factors,
                                                                         mode='B')
        gdf_evaluated_pflaster_platten_n = evaluate_gdf_pflaster_platten(gdf_pflaster_platten_n,
                                                                         normalization_factors=normalization_factors,
                                                                         weighting_factors=weighting_factors,
                                                                         mode='N')

        gdfs_evaluated.extend([gdf_evaluated_pflaster_platten_a,
                               gdf_evaluated_pflaster_platten_b,
                               gdf_evaluated_pflaster_platten_n])

    gdf_evaluated = concatenate_gdfs(gdfs_evaluated)

    return gdf_evaluated
