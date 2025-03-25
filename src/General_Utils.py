# @author: Roß, Alexander - Fachbereich Elektrotechnik, Westfälische Hochschule Gelsenkirchen
# @coauthor: Maryniak, Marius - Fachbereich Elektrotechnik, Westfälische Hochschule Gelsenkirchen

import multiprocessing
import os
import time

import cv2
import geopandas as gpd
import numpy as np
import onnxruntime
import pandas
import rasterio
from rasterio.transform import Affine
from shapely.geometry import LineString, Polygon
from scipy.ndimage import gaussian_filter
from scipy.spatial.transform import Rotation

from src import Ortho_Creation_Utils
from src import Postprocessing_Utils
from src.GSC_Utils import LaserData
from src.GSC_Utils import file_id
from src.Ortho_Creation_Utils import subsample_ortho, tile_buffer
from src.Planeness_Utils import create_planeness_sampling_points, sample_points

file_res = 50 / 5120
filter_sigma = 1 // (file_res * 5)

roadmarkings_segmenter = onnxruntime.InferenceSession("data/models/abs_m.onnx")
roadmarkings_segmenter_input_name = roadmarkings_segmenter.get_inputs()[0].name

substance_segmenter = onnxruntime.InferenceSession("data/models/abs_z.onnx")
substance_segmenter_input_name_rgbi = substance_segmenter.get_inputs()[0].name
substance_segmenter_input_name_height = substance_segmenter.get_inputs()[1].name


def get_from_nested_dict(key_list, nested_dict):
    """
    Get value from nested dict. returns None for nonexistent keys.

    :param list key_list: List representing the chain of keys to get to the desired value.
    :param nested_dict: Nested dict to get from.
    :return: Value from nested dict or None if keys are not present.
    """
    value = nested_dict
    for key in key_list[:-1]:
        value = value.get(key, {})
    value = value.get(key_list[-1])
    return value


def parse_required(key_list, config):
    """
    Parse required key-list in nested config-dict. Raises Error if key is missing.

    :param list key_list: List representing the chain of keys to get the required value.
    :param config: Nested config-dict.
    :return: Value from nested config-dict.
    """
    value = get_from_nested_dict(key_list, config)
    if value is None:
        raise ValueError(f"{' : '.join(map(str, key_list))} muss in Konfiguration spezifiziert werden!")

    return value


def parse_optional(key_list, config, default=None):
    """
    Parse optional key-list in nested config-dict. Returns default if key is missing.

    :param list key_list: List representing the chain of keys to get the required value.
    :param config: Nested config-dict.
    :param default: Default value returned if key is missing.
    :return: Value from nested config-dict or default.
    """
    value = get_from_nested_dict(key_list, config)
    if value is None:
        value = default

    return value


def parse_recording_points(config):
    """
    Parse shapefile info from nested config-dict.

    :param config: Nested config-dict.
    :return: Shapefile info as dictionary.
    :rtype: dict or None
    """
    path = parse_required(["EINGANGSDATEN", "PANORAMAS", "AUFNAHMEPUNKTE", "PFAD"], config)
    id_base = parse_optional(["EINGANGSDATEN", "PANORAMAS", "AUFNAHMEPUNKTE", "ID_BASIS"], config, default=36)
    id_key = parse_required(["EINGANGSDATEN", "PANORAMAS", "AUFNAHMEPUNKTE", "FELDER", "ID"], config)
    recording_time_key = parse_required(["EINGANGSDATEN", "PANORAMAS", "AUFNAHMEPUNKTE", "FELDER", "AUFNAHMEZEITPUNKT"], config)
    recording_direction_key = parse_required(["EINGANGSDATEN", "PANORAMAS", "AUFNAHMEPUNKTE", "FELDER", "AUFNAHMERICHTUNG"],
                                             config)
    recording_height_key = parse_required(["EINGANGSDATEN", "PANORAMAS", "AUFNAHMEPUNKTE", "FELDER", "AUFNAHMEHOEHE"], config)
    image_path_key = parse_optional(["EINGANGSDATEN", "PANORAMAS", "AUFNAHMEPUNKTE", "FELDER", "PANORAMA_PFAD"], config)
    mask_path_key = parse_optional(["EINGANGSDATEN", "PANORAMAS", "AUFNAHMEPUNKTE", "FELDER", "MASKEN_PFAD"], config)
    pitch_key = parse_optional(["EINGANGSDATEN", "PANORAMAS", "AUFNAHMEPUNKTE", "FELDER", "AUFNAHMEPITCH"], config)
    roll_key = parse_optional(["EINGANGSDATEN", "PANORAMAS", "AUFNAHMEPUNKTE", "FELDER", "AUFNAHMEROLL"], config)

    if path is None:
        return None
    else:
        return {"path": path,
                "id_base": id_base,
                "id_key": id_key,
                "recording_time_key": recording_time_key,
                "recording_direction_key": recording_direction_key,
                "recording_height_key": recording_height_key,
                "image_path_key": image_path_key,
                "mask_path_key": mask_path_key,
                "pitch_key": pitch_key,
                "roll_key": roll_key}


def parse_weighting_factors(config):
    """
    Parse weighting-facotrs from nested config-dict.

    :param config: Nested config-dict.
    :return: Array of weighting facotrs.
    :rtype: np.ndarray
    """
    gut = parse_optional(["NACHVERARBEITUNG", "STRASSENMARKIERUNGEN", "GEWICHTUNGSFAKTOREN_ZUSTAND", "GUT"], config, default=1)
    mittel = parse_optional(["NACHVERARBEITUNG", "STRASSENMARKIERUNGEN", "GEWICHTUNGSFAKTOREN_ZUSTAND", "MITTEL"], config, default=1)
    schlecht = parse_optional(["NACHVERARBEITUNG", "STRASSENMARKIERUNGEN", "GEWICHTUNGSFAKTOREN_ZUSTAND", "SCHLECHT"], config, default=1)

    linie_duenn = parse_optional(["NACHVERARBEITUNG", "STRASSENMARKIERUNGEN", "GEWICHTUNGSFAKTOREN_TYP", "LINIE_DUENN"], config, default=1)
    linie_duenn_gestrichelt = parse_optional(["NACHVERARBEITUNG", "STRASSENMARKIERUNGEN", "GEWICHTUNGSFAKTOREN_TYP",
                                              "LINIE_DUENN_GESTRICHELT"], config, default=1)
    linie_breit = parse_optional(["NACHVERARBEITUNG", "STRASSENMARKIERUNGEN", "GEWICHTUNGSFAKTOREN_TYP", "LINIE_BREIT"], config, default=1)
    linie_breit_gestrichelt = parse_optional(["NACHVERARBEITUNG", "GEWICHTUNGSFAKTOREN_TYP", "LINIE_BREIT_GESTRICHELT"],
                                             config, default=1)
    linie_extrabreit = parse_optional(["NACHVERARBEITUNG", "STRASSENMARKIERUNGEN", "GEWICHTUNGSFAKTOREN_TYP", "LINIE_EXTRABREIT"],
                                      config, default=1)
    pfeile = parse_optional(["NACHVERARBEITUNG", "STRASSENMARKIERUNGEN", "GEWICHTUNGSFAKTOREN_TYP", "PFEILE"], config, default=1)
    symbole = parse_optional(["NACHVERARBEITUNG", "STRASSENMARKIERUNGEN", "GEWICHTUNGSFAKTOREN_TYP", "SYMBOLE"], config, default=1)
    sperrflaechen = parse_optional(["NACHVERARBEITUNG", "STRASSENMARKIERUNGEN", "GEWICHTUNGSFAKTOREN_TYP", "SPERRFLAECHEN"],
                                   config, default=1)

    condition_factors = np.array([[gut, mittel, schlecht]])

    type_factors = np.array([[linie_duenn,
                              linie_duenn_gestrichelt,
                              linie_breit,
                              linie_breit_gestrichelt,
                              linie_extrabreit,
                              0,
                              pfeile,
                              symbole,
                              sperrflaechen]])

    factors = np.matmul(condition_factors.T, type_factors)

    return factors


def parse_thresholds(config):
    """
    Parse postprocessing thresholds from nested config-dict.

    :param config: Nested config-dict.
    :return: Dicr with thresholds.
    :rtype: dict
    """
    tiles_schlecht_blocking_gut = parse_optional(["NACHVERARBEITUNG", "STRASSENMARKIERUNGEN", "SCHWELLENWERTE", "KACHELN",
                                                  "SCHLECHT_NICHT_GUT"],
                                                 config, default=0.02)
    tiles_schlecht = parse_optional(["NACHVERARBEITUNG", "STRASSENMARKIERUNGEN", "SCHWELLENWERTE", "KACHELN", "SCHLECHT"],
                                    config, default=0.3)
    tiles_mittel = parse_optional(["NACHVERARBEITUNG", "SCHWELLENWERTE", "KACHELN", "MITTEL"],
                                  config, default=0.2)

    markings_schlecht = parse_optional(["NACHVERARBEITUNG", "STRASSENMARKIERUNGEN", "SCHWELLENWERTE", "EINZELMARKIERUNGEN", "SCHLECHT"],
                                       config, default=0.25)
    markings_mittel = parse_optional(["NACHVERARBEITUNG", "STRASSENMARKIERUNGEN", "SCHWELLENWERTE", "EINZELMARKIERUNGEN", "MITTEL"],
                                     config, default=0.25)

    threshold_dict = {"tiles_schlecht_blocking_gut": tiles_schlecht_blocking_gut,
                      "tiles_schlecht": tiles_schlecht,
                      "tiles_mittel": tiles_mittel,
                      "markings_schlecht": markings_schlecht,
                      "markings_mittel": markings_mittel}

    return threshold_dict


def parse_roadcondition_normalization_factors_asphalt(config):
    sm4l_m_a_1_5 = parse_optional(["NACHVERARBEITUNG", "STRASSENZUSTAND", "NORMIERUNGSFAKTOREN", "ASPHALT",
                                   "SM4L_M", "FUNKTIONSKLASSE_A", "1_5_WERT"], config, default=4.)
    sm4l_m_a_warn = parse_optional(["NACHVERARBEITUNG", "STRASSENZUSTAND", "NORMIERUNGSFAKTOREN", "ASPHALT",
                                    "SM4L_M", "FUNKTIONSKLASSE_A", "WARNWERT"], config, default=12.)
    sm4l_m_a_schwelle = parse_optional(["NACHVERARBEITUNG", "STRASSENZUSTAND", "NORMIERUNGSFAKTOREN", "ASPHALT",
                                        "SM4L_M", "FUNKTIONSKLASSE_A", "SCHWELLENWERT"], config, default=16.)
    sm4l_m_b_1_5 = parse_optional(["NACHVERARBEITUNG", "STRASSENZUSTAND", "NORMIERUNGSFAKTOREN", "ASPHALT",
                                   "SM4L_M", "FUNKTIONSKLASSE_B", "1_5_WERT"], config, default=4.)
    sm4l_m_b_warn = parse_optional(["NACHVERARBEITUNG", "STRASSENZUSTAND", "NORMIERUNGSFAKTOREN", "ASPHALT",
                                    "SM4L_M", "FUNKTIONSKLASSE_B", "WARNWERT"], config, default=16.)
    sm4l_m_b_schwelle = parse_optional(["NACHVERARBEITUNG", "STRASSENZUSTAND", "NORMIERUNGSFAKTOREN", "ASPHALT",
                                        "SM4L_M", "FUNKTIONSKLASSE_B", "SCHWELLENWERT"], config, default=25.)

    sm4l_a_a_1_5 = parse_optional(["NACHVERARBEITUNG", "STRASSENZUSTAND", "NORMIERUNGSFAKTOREN", "ASPHALT",
                                   "SM4L_A", "FUNKTIONSKLASSE_A", "1_5_WERT"], config, default=1.)
    sm4l_a_a_warn = parse_optional(["NACHVERARBEITUNG", "STRASSENZUSTAND", "NORMIERUNGSFAKTOREN", "ASPHALT",
                                    "SM4L_A", "FUNKTIONSKLASSE_A", "WARNWERT"], config, default=2.5)
    sm4l_a_a_schwelle = parse_optional(["NACHVERARBEITUNG", "STRASSENZUSTAND", "NORMIERUNGSFAKTOREN", "ASPHALT",
                                        "SM4L_A", "FUNKTIONSKLASSE_A", "SCHWELLENWERT"], config, default=3.5)
    sm4l_a_b_1_5 = parse_optional(["NACHVERARBEITUNG", "STRASSENZUSTAND", "NORMIERUNGSFAKTOREN", "ASPHALT",
                                   "SM4L_A", "FUNKTIONSKLASSE_B", "1_5_WERT"], config, default=1.)
    sm4l_a_b_warn = parse_optional(["NACHVERARBEITUNG", "STRASSENZUSTAND", "NORMIERUNGSFAKTOREN", "ASPHALT",
                                    "SM4L_A", "FUNKTIONSKLASSE_B", "WARNWERT"], config, default=3.5)
    sm4l_a_b_schwelle = parse_optional(["NACHVERARBEITUNG", "STRASSENZUSTAND", "NORMIERUNGSFAKTOREN", "ASPHALT",
                                        "SM4L_A", "FUNKTIONSKLASSE_B", "SCHWELLENWERT"], config, default=5.)

    mspt_a_1_5 = parse_optional(["NACHVERARBEITUNG", "STRASSENZUSTAND", "NORMIERUNGSFAKTOREN", "ASPHALT",
                                 "MSPT", "FUNKTIONSKLASSE_A", "1_5_WERT"], config, default=4.)
    mspt_a_warn = parse_optional(["NACHVERARBEITUNG", "STRASSENZUSTAND", "NORMIERUNGSFAKTOREN", "ASPHALT",
                                  "MSPT", "FUNKTIONSKLASSE_A", "WARNWERT"], config, default=15.)
    mspt_a_schwelle = parse_optional(["NACHVERARBEITUNG", "STRASSENZUSTAND", "NORMIERUNGSFAKTOREN", "ASPHALT",
                                      "MSPT", "FUNKTIONSKLASSE_A", "SCHWELLENWERT"], config, default=25.)
    mspt_b_1_5 = parse_optional(["NACHVERARBEITUNG", "STRASSENZUSTAND", "NORMIERUNGSFAKTOREN", "ASPHALT",
                                 "MSPT", "FUNKTIONSKLASSE_B", "1_5_WERT"], config, default=4.)
    mspt_b_warn = parse_optional(["NACHVERARBEITUNG", "STRASSENZUSTAND", "NORMIERUNGSFAKTOREN", "ASPHALT",
                                  "MSPT", "FUNKTIONSKLASSE_B", "WARNWERT"], config, default=15.)
    mspt_b_schwelle = parse_optional(["NACHVERARBEITUNG", "STRASSENZUSTAND", "NORMIERUNGSFAKTOREN", "ASPHALT",
                                      "MSPT", "FUNKTIONSKLASSE_B", "SCHWELLENWERT"], config, default=25.)

    msph_a_1_5 = parse_optional(["NACHVERARBEITUNG", "STRASSENZUSTAND", "NORMIERUNGSFAKTOREN", "ASPHALT",
                                 "MSPH", "FUNKTIONSKLASSE_A", "1_5_WERT"], config, default=.1)
    msph_a_warn = parse_optional(["NACHVERARBEITUNG", "STRASSENZUSTAND", "NORMIERUNGSFAKTOREN", "ASPHALT",
                                  "MSPH", "FUNKTIONSKLASSE_A", "WARNWERT"], config, default=4.)
    msph_a_schwelle = parse_optional(["NACHVERARBEITUNG", "STRASSENZUSTAND", "NORMIERUNGSFAKTOREN", "ASPHALT",
                                      "MSPH", "FUNKTIONSKLASSE_A", "SCHWELLENWERT"], config, default=6.)
    msph_b_1_5 = parse_optional(["NACHVERARBEITUNG", "STRASSENZUSTAND", "NORMIERUNGSFAKTOREN", "ASPHALT",
                                 "MSPH", "FUNKTIONSKLASSE_B", "1_5_WERT"], config, default=.1)
    msph_b_warn = parse_optional(["NACHVERARBEITUNG", "STRASSENZUSTAND", "NORMIERUNGSFAKTOREN", "ASPHALT",
                                  "MSPH", "FUNKTIONSKLASSE_B", "WARNWERT"], config, default=4.)
    msph_b_schwelle = parse_optional(["NACHVERARBEITUNG", "STRASSENZUSTAND", "NORMIERUNGSFAKTOREN", "ASPHALT",
                                      "MSPH", "FUNKTIONSKLASSE_B", "SCHWELLENWERT"], config, default=6.)

    riss_a_1_5 = parse_optional(["NACHVERARBEITUNG", "STRASSENZUSTAND", "NORMIERUNGSFAKTOREN", "ASPHALT",
                                 "RISS", "FUNKTIONSKLASSE_A", "1_5_WERT"], config, default=1)
    riss_a_warn = parse_optional(["NACHVERARBEITUNG", "STRASSENZUSTAND", "NORMIERUNGSFAKTOREN", "ASPHALT",
                                  "RISS", "FUNKTIONSKLASSE_A", "WARNWERT"], config, default=15)
    riss_a_schwelle = parse_optional(["NACHVERARBEITUNG", "STRASSENZUSTAND", "NORMIERUNGSFAKTOREN", "ASPHALT",
                                      "RISS", "FUNKTIONSKLASSE_A", "SCHWELLENWERT"], config, default=25)
    riss_b_1_5 = parse_optional(["NACHVERARBEITUNG", "STRASSENZUSTAND", "NORMIERUNGSFAKTOREN", "ASPHALT",
                                 "RISS", "FUNKTIONSKLASSE_B", "1_5_WERT"], config, default=5)
    riss_b_warn = parse_optional(["NACHVERARBEITUNG", "STRASSENZUSTAND", "NORMIERUNGSFAKTOREN", "ASPHALT",
                                  "RISS", "FUNKTIONSKLASSE_B", "WARNWERT"], config, default=20)
    riss_b_schwelle = parse_optional(["NACHVERARBEITUNG", "STRASSENZUSTAND", "NORMIERUNGSFAKTOREN", "ASPHALT",
                                      "RISS", "FUNKTIONSKLASSE_B", "SCHWELLENWERT"], config, default=33)
    riss_n_1_5 = parse_optional(["NACHVERARBEITUNG", "STRASSENZUSTAND", "NORMIERUNGSFAKTOREN", "ASPHALT",
                                 "RISS", "FUNKTIONSKLASSE_N", "1_5_WERT"], config, default=5)
    riss_n_warn = parse_optional(["NACHVERARBEITUNG", "STRASSENZUSTAND", "NORMIERUNGSFAKTOREN", "ASPHALT",
                                  "RISS", "FUNKTIONSKLASSE_N", "WARNWERT"], config, default=20)
    riss_n_schwelle = parse_optional(["NACHVERARBEITUNG", "STRASSENZUSTAND", "NORMIERUNGSFAKTOREN", "ASPHALT",
                                      "RISS", "FUNKTIONSKLASSE_N", "SCHWELLENWERT"], config, default=33)

    afli_a_1_5 = parse_optional(["NACHVERARBEITUNG", "STRASSENZUSTAND", "NORMIERUNGSFAKTOREN", "ASPHALT",
                                 "AFLI", "FUNKTIONSKLASSE_A", "1_5_WERT"], config, default=1)
    afli_a_warn = parse_optional(["NACHVERARBEITUNG", "STRASSENZUSTAND", "NORMIERUNGSFAKTOREN", "ASPHALT",
                                  "AFLI", "FUNKTIONSKLASSE_A", "WARNWERT"], config, default=15)
    afli_a_schwelle = parse_optional(["NACHVERARBEITUNG", "STRASSENZUSTAND", "NORMIERUNGSFAKTOREN", "ASPHALT",
                                      "AFLI", "FUNKTIONSKLASSE_A", "SCHWELLENWERT"], config, default=25)
    afli_b_1_5 = parse_optional(["NACHVERARBEITUNG", "STRASSENZUSTAND", "NORMIERUNGSFAKTOREN", "ASPHALT",
                                 "AFLI", "FUNKTIONSKLASSE_B", "1_5_WERT"], config, default=5)
    afli_b_warn = parse_optional(["NACHVERARBEITUNG", "STRASSENZUSTAND", "NORMIERUNGSFAKTOREN", "ASPHALT",
                                  "AFLI", "FUNKTIONSKLASSE_B", "WARNWERT"], config, default=20)
    afli_b_schwelle = parse_optional(["NACHVERARBEITUNG", "STRASSENZUSTAND", "NORMIERUNGSFAKTOREN", "ASPHALT",
                                      "AFLI", "FUNKTIONSKLASSE_B", "SCHWELLENWERT"], config, default=33)
    afli_n_1_5 = parse_optional(["NACHVERARBEITUNG", "STRASSENZUSTAND", "NORMIERUNGSFAKTOREN", "ASPHALT",
                                 "AFLI", "FUNKTIONSKLASSE_N", "1_5_WERT"], config, default=5)
    afli_n_warn = parse_optional(["NACHVERARBEITUNG", "STRASSENZUSTAND", "NORMIERUNGSFAKTOREN", "ASPHALT",
                                  "AFLI", "FUNKTIONSKLASSE_N", "WARNWERT"], config, default=20)
    afli_n_schwelle = parse_optional(["NACHVERARBEITUNG", "STRASSENZUSTAND", "NORMIERUNGSFAKTOREN", "ASPHALT",
                                      "AFLI", "FUNKTIONSKLASSE_N", "SCHWELLENWERT"], config, default=33)

    ofs_a_1_5 = parse_optional(["NACHVERARBEITUNG", "STRASSENZUSTAND", "NORMIERUNGSFAKTOREN", "ASPHALT",
                                "OFS", "FUNKTIONSKLASSE_A", "1_5_WERT"], config, default=1)
    ofs_a_warn = parse_optional(["NACHVERARBEITUNG", "STRASSENZUSTAND", "NORMIERUNGSFAKTOREN", "ASPHALT",
                                 "OFS", "FUNKTIONSKLASSE_A", "WARNWERT"], config, default=15)
    ofs_a_schwelle = parse_optional(["NACHVERARBEITUNG", "STRASSENZUSTAND", "NORMIERUNGSFAKTOREN", "ASPHALT",
                                     "OFS", "FUNKTIONSKLASSE_A", "SCHWELLENWERT"], config, default=25)
    ofs_b_1_5 = parse_optional(["NACHVERARBEITUNG", "STRASSENZUSTAND", "NORMIERUNGSFAKTOREN", "ASPHALT",
                                "OFS", "FUNKTIONSKLASSE_B", "1_5_WERT"], config, default=10)
    ofs_b_warn = parse_optional(["NACHVERARBEITUNG", "STRASSENZUSTAND", "NORMIERUNGSFAKTOREN", "ASPHALT",
                                 "OFS", "FUNKTIONSKLASSE_B", "WARNWERT"], config, default=33)
    ofs_b_schwelle = parse_optional(["NACHVERARBEITUNG", "STRASSENZUSTAND", "NORMIERUNGSFAKTOREN", "ASPHALT",
                                     "OFS", "FUNKTIONSKLASSE_B", "SCHWELLENWERT"], config, default=50)
    ofs_n_1_5 = parse_optional(["NACHVERARBEITUNG", "STRASSENZUSTAND", "NORMIERUNGSFAKTOREN", "ASPHALT",
                                "OFS", "FUNKTIONSKLASSE_N", "1_5_WERT"], config, default=10)
    ofs_n_warn = parse_optional(["NACHVERARBEITUNG", "STRASSENZUSTAND", "NORMIERUNGSFAKTOREN", "ASPHALT",
                                 "OFS", "FUNKTIONSKLASSE_N", "WARNWERT"], config, default=33)
    ofs_n_schwelle = parse_optional(["NACHVERARBEITUNG", "STRASSENZUSTAND", "NORMIERUNGSFAKTOREN", "ASPHALT",
                                     "OFS", "FUNKTIONSKLASSE_N", "SCHWELLENWERT"], config, default=50)

    normalization_factors = {"SM4L_M": {"A": (sm4l_m_a_1_5, sm4l_m_a_warn, sm4l_m_a_schwelle),
                                        "B": (sm4l_m_b_1_5, sm4l_m_b_warn, sm4l_m_b_schwelle)},
                             "SM4L_A": {"A": (sm4l_a_a_1_5, sm4l_a_a_warn, sm4l_a_a_schwelle),
                                        "B": (sm4l_a_b_1_5, sm4l_a_b_warn, sm4l_a_b_schwelle)},
                             "MSPT": {"A": (mspt_a_1_5, mspt_a_warn, mspt_a_schwelle),
                                      "B": (mspt_b_1_5, mspt_b_warn, mspt_b_schwelle)},
                             "MSPH": {"A": (msph_a_1_5, msph_a_warn, msph_a_schwelle),
                                      "B": (msph_b_1_5, msph_b_warn, msph_b_schwelle)},
                             "RISS": {"A": (riss_a_1_5, riss_a_warn, riss_a_schwelle),
                                      "B": (riss_b_1_5, riss_b_warn, riss_b_schwelle),
                                      "N": (riss_n_1_5, riss_n_warn, riss_n_schwelle)},
                             "AFLI": {"A": (afli_a_1_5, afli_a_warn, afli_a_schwelle),
                                      "B": (afli_b_1_5, afli_b_warn, afli_b_schwelle),
                                      "N": (afli_n_1_5, afli_n_warn, afli_n_schwelle)},
                             "OFS": {"A": (ofs_a_1_5, ofs_a_warn, ofs_a_schwelle),
                                     "B": (ofs_b_1_5, ofs_b_warn, ofs_b_schwelle),
                                     "N": (ofs_n_1_5, ofs_n_warn, ofs_n_schwelle)}}

    return normalization_factors


def parse_roadcondition_normalization_factors_pflaster_platten(config):
    sm4l_m_a_1_5 = parse_optional(["NACHVERARBEITUNG", "STRASSENZUSTAND", "NORMIERUNGSFAKTOREN", "PFLASTER_PLATTEN",
                                   "SM4L_M", "FUNKTIONSKLASSE_A", "1_5_WERT"], config, default=4.)
    sm4l_m_a_warn = parse_optional(["NACHVERARBEITUNG", "STRASSENZUSTAND", "NORMIERUNGSFAKTOREN", "PFLASTER_PLATTEN",
                                    "SM4L_M", "FUNKTIONSKLASSE_A", "WARNWERT"], config, default=12.)
    sm4l_m_a_schwelle = parse_optional(["NACHVERARBEITUNG", "STRASSENZUSTAND", "NORMIERUNGSFAKTOREN", "PFLASTER_PLATTEN",
                                        "SM4L_M", "FUNKTIONSKLASSE_A", "SCHWELLENWERT"], config, default=16.)
    sm4l_m_b_1_5 = parse_optional(["NACHVERARBEITUNG", "STRASSENZUSTAND", "NORMIERUNGSFAKTOREN", "PFLASTER_PLATTEN",
                                   "SM4L_M", "FUNKTIONSKLASSE_B", "1_5_WERT"], config, default=4.)
    sm4l_m_b_warn = parse_optional(["NACHVERARBEITUNG", "STRASSENZUSTAND", "NORMIERUNGSFAKTOREN", "PFLASTER_PLATTEN",
                                    "SM4L_M", "FUNKTIONSKLASSE_B", "WARNWERT"], config, default=16.)
    sm4l_m_b_schwelle = parse_optional(["NACHVERARBEITUNG", "STRASSENZUSTAND", "NORMIERUNGSFAKTOREN", "PFLASTER_PLATTEN",
                                        "SM4L_M", "FUNKTIONSKLASSE_B", "SCHWELLENWERT"], config, default=25.)

    sm4l_a_a_1_5 = parse_optional(["NACHVERARBEITUNG", "STRASSENZUSTAND", "NORMIERUNGSFAKTOREN", "PFLASTER_PLATTEN",
                                   "SM4L_A", "FUNKTIONSKLASSE_A", "1_5_WERT"], config, default=1.)
    sm4l_a_a_warn = parse_optional(["NACHVERARBEITUNG", "STRASSENZUSTAND", "NORMIERUNGSFAKTOREN", "PFLASTER_PLATTEN",
                                    "SM4L_A", "FUNKTIONSKLASSE_A", "WARNWERT"], config, default=2.5)
    sm4l_a_a_schwelle = parse_optional(["NACHVERARBEITUNG", "STRASSENZUSTAND", "NORMIERUNGSFAKTOREN", "PFLASTER_PLATTEN",
                                        "SM4L_A", "FUNKTIONSKLASSE_A", "SCHWELLENWERT"], config, default=3.5)
    sm4l_a_b_1_5 = parse_optional(["NACHVERARBEITUNG", "STRASSENZUSTAND", "NORMIERUNGSFAKTOREN", "PFLASTER_PLATTEN",
                                   "SM4L_A", "FUNKTIONSKLASSE_B", "1_5_WERT"], config, default=1.)
    sm4l_a_b_warn = parse_optional(["NACHVERARBEITUNG", "STRASSENZUSTAND", "NORMIERUNGSFAKTOREN", "PFLASTER_PLATTEN",
                                    "SM4L_A", "FUNKTIONSKLASSE_B", "WARNWERT"], config, default=3.5)
    sm4l_a_b_schwelle = parse_optional(["NACHVERARBEITUNG", "STRASSENZUSTAND", "NORMIERUNGSFAKTOREN", "PFLASTER_PLATTEN",
                                        "SM4L_A", "FUNKTIONSKLASSE_B", "SCHWELLENWERT"], config, default=5.)

    mspt_a_1_5 = parse_optional(["NACHVERARBEITUNG", "STRASSENZUSTAND", "NORMIERUNGSFAKTOREN", "PFLASTER_PLATTEN",
                                 "MSPT", "FUNKTIONSKLASSE_A", "1_5_WERT"], config, default=4.)
    mspt_a_warn = parse_optional(["NACHVERARBEITUNG", "STRASSENZUSTAND", "NORMIERUNGSFAKTOREN", "PFLASTER_PLATTEN",
                                  "MSPT", "FUNKTIONSKLASSE_A", "WARNWERT"], config, default=15.)
    mspt_a_schwelle = parse_optional(["NACHVERARBEITUNG", "STRASSENZUSTAND", "NORMIERUNGSFAKTOREN", "PFLASTER_PLATTEN",
                                      "MSPT", "FUNKTIONSKLASSE_A", "SCHWELLENWERT"], config, default=25.)
    mspt_b_1_5 = parse_optional(["NACHVERARBEITUNG", "STRASSENZUSTAND", "NORMIERUNGSFAKTOREN", "PFLASTER_PLATTEN",
                                 "MSPT", "FUNKTIONSKLASSE_B", "1_5_WERT"], config, default=4.)
    mspt_b_warn = parse_optional(["NACHVERARBEITUNG", "STRASSENZUSTAND", "NORMIERUNGSFAKTOREN", "PFLASTER_PLATTEN",
                                  "MSPT", "FUNKTIONSKLASSE_B", "WARNWERT"], config, default=15.)
    mspt_b_schwelle = parse_optional(["NACHVERARBEITUNG", "STRASSENZUSTAND", "NORMIERUNGSFAKTOREN", "PFLASTER_PLATTEN",
                                      "MSPT", "FUNKTIONSKLASSE_B", "SCHWELLENWERT"], config, default=25.)

    msph_a_1_5 = parse_optional(["NACHVERARBEITUNG", "STRASSENZUSTAND", "NORMIERUNGSFAKTOREN", "PFLASTER_PLATTEN",
                                 "MSPH", "FUNKTIONSKLASSE_A", "1_5_WERT"], config, default=.1)
    msph_a_warn = parse_optional(["NACHVERARBEITUNG", "STRASSENZUSTAND", "NORMIERUNGSFAKTOREN", "PFLASTER_PLATTEN",
                                  "MSPH", "FUNKTIONSKLASSE_A", "WARNWERT"], config, default=4.)
    msph_a_schwelle = parse_optional(["NACHVERARBEITUNG", "STRASSENZUSTAND", "NORMIERUNGSFAKTOREN", "PFLASTER_PLATTEN",
                                      "MSPH", "FUNKTIONSKLASSE_A", "SCHWELLENWERT"], config, default=6.)
    msph_b_1_5 = parse_optional(["NACHVERARBEITUNG", "STRASSENZUSTAND", "NORMIERUNGSFAKTOREN", "PFLASTER_PLATTEN",
                                 "MSPH", "FUNKTIONSKLASSE_B", "1_5_WERT"], config, default=.1)
    msph_b_warn = parse_optional(["NACHVERARBEITUNG", "STRASSENZUSTAND", "NORMIERUNGSFAKTOREN", "PFLASTER_PLATTEN",
                                  "MSPH", "FUNKTIONSKLASSE_B", "WARNWERT"], config, default=4.)
    msph_b_schwelle = parse_optional(["NACHVERARBEITUNG", "STRASSENZUSTAND", "NORMIERUNGSFAKTOREN", "PFLASTER_PLATTEN",
                                      "MSPH", "FUNKTIONSKLASSE_B", "SCHWELLENWERT"], config, default=6.)

    ofs_a_1_5 = parse_optional(["NACHVERARBEITUNG", "STRASSENZUSTAND", "NORMIERUNGSFAKTOREN", "PFLASTER_PLATTEN",
                                "OFS", "FUNKTIONSKLASSE_A", "1_5_WERT"], config, default=1)
    ofs_a_warn = parse_optional(["NACHVERARBEITUNG", "STRASSENZUSTAND", "NORMIERUNGSFAKTOREN", "PFLASTER_PLATTEN",
                                 "OFS", "FUNKTIONSKLASSE_A", "WARNWERT"], config, default=15)
    ofs_a_schwelle = parse_optional(["POSTPROCESSING", "STRASSENZUSTAND", "NORMIERUNGSFAKTOREN", "PFLASTER_PLATTEN",
                                     "OFS", "FUNKTIONSKLASSE_A", "SCHWELLENWERT"], config, default=25)
    ofs_b_1_5 = parse_optional(["POSTPROCESSING", "STRASSENZUSTAND", "NORMIERUNGSFAKTOREN", "PFLASTER_PLATTEN",
                                "OFS", "FUNKTIONSKLASSE_B", "1_5_WERT"], config, default=5)
    ofs_b_warn = parse_optional(["POSTPROCESSING", "STRASSENZUSTAND", "NORMIERUNGSFAKTOREN", "PFLASTER_PLATTEN",
                                 "OFS", "FUNKTIONSKLASSE_B", "WARNWERT"], config, default=25)
    ofs_b_schwelle = parse_optional(["POSTPROCESSING", "STRASSENZUSTAND", "NORMIERUNGSFAKTOREN", "PFLASTER_PLATTEN",
                                     "OFS", "FUNKTIONSKLASSE_B", "SCHWELLENWERT"], config, default=40)
    ofs_n_1_5 = parse_optional(["POSTPROCESSING", "STRASSENZUSTAND", "NORMIERUNGSFAKTOREN", "PFLASTER_PLATTEN",
                                "OFS", "FUNKTIONSKLASSE_N", "1_5_WERT"], config, default=5)
    ofs_n_warn = parse_optional(["POSTPROCESSING", "STRASSENZUSTAND", "NORMIERUNGSFAKTOREN", "PFLASTER_PLATTEN",
                                 "OFS", "FUNKTIONSKLASSE_N", "WARNWERT"], config, default=25)
    ofs_n_schwelle = parse_optional(["POSTPROCESSING", "STRASSENZUSTAND", "NORMIERUNGSFAKTOREN", "PFLASTER_PLATTEN",
                                     "OFS", "FUNKTIONSKLASSE_N", "SCHWELLENWERT"], config, default=40)

    normalization_factors = {"SM4L_M": {"A": (sm4l_m_a_1_5, sm4l_m_a_warn, sm4l_m_a_schwelle),
                                        "B": (sm4l_m_b_1_5, sm4l_m_b_warn, sm4l_m_b_schwelle)},
                             "SM4L_A": {"A": (sm4l_a_a_1_5, sm4l_a_a_warn, sm4l_a_a_schwelle),
                                        "B": (sm4l_a_b_1_5, sm4l_a_b_warn, sm4l_a_b_schwelle)},
                             "MSPT": {"A": (mspt_a_1_5, mspt_a_warn, mspt_a_schwelle),
                                      "B": (mspt_b_1_5, mspt_b_warn, mspt_b_schwelle)},
                             "MSPH": {"A": (msph_a_1_5, msph_a_warn, msph_a_schwelle),
                                      "B": (msph_b_1_5, msph_b_warn, msph_b_schwelle)},
                             "OFS": {"A": (ofs_a_1_5, ofs_a_warn, ofs_a_schwelle),
                                     "B": (ofs_b_1_5, ofs_b_warn, ofs_b_schwelle),
                                     "N": (ofs_n_1_5, ofs_n_warn, ofs_n_schwelle)}}

    return normalization_factors


def parse_roadcondition_weighting_factors_asphalt(config):
    geb_zwsm4l = parse_optional(["POSTPROCESSING", "STRASSENZUSTAND", "GEWICHTUNGSFAKTOREN", "ASPHALT",
                                 "GEBRAUCHSWERT", "ZWSM4L"], config, default=.25)
    geb_zwmspt = parse_optional(["POSTPROCESSING", "STRASSENZUSTAND", "GEWICHTUNGSFAKTOREN", "ASPHALT",
                                 "GEBRAUCHSWERT", "ZWMSPT"], config, default=.5)
    geb_zwmsph = parse_optional(["POSTPROCESSING", "STRASSENZUSTAND", "GEWICHTUNGSFAKTOREN", "ASPHALT",
                                 "GEBRAUCHSWERT", "ZWMSPH"], config, default=.25)

    sub_zwriss = parse_optional(["POSTPROCESSING", "STRASSENZUSTAND", "GEWICHTUNGSFAKTOREN", "ASPHALT",
                                 "SUBSTANZWERT", "ZWRISS"], config, default=.56)
    sub_zwafli = parse_optional(["POSTPROCESSING", "STRASSENZUSTAND", "GEWICHTUNGSFAKTOREN", "ASPHALT",
                                 "SUBSTANZWERT", "ZWAFLI"], config, default=.31)
    sub_zwofs = parse_optional(["POSTPROCESSING", "STRASSENZUSTAND", "GEWICHTUNGSFAKTOREN", "ASPHALT",
                                "SUBSTANZWERT", "ZWOFS"], config, default=.13)

    weighting_factors = {"TWGEB": {"ZWSM4L": geb_zwsm4l,
                                   "ZWMSPT": geb_zwmspt,
                                   "ZWMSPH": geb_zwmsph},
                         "TWSUB": {"ZWRISS": sub_zwriss,
                                   "ZWAFLI": sub_zwafli,
                                   "ZWOFS": sub_zwofs}}

    return weighting_factors


def parse_roadcondition_weighting_factors_pflaster_platten(config):
    geb_zwsm4l = parse_optional(["POSTPROCESSING", "STRASSENZUSTAND", "GEWICHTUNGSFAKTOREN", "PFLASTER_PLATTEN",
                                 "GEBRAUCHSWERT", "ZWSM4L"], config, default=.25)
    geb_zwmspt = parse_optional(["POSTPROCESSING", "STRASSENZUSTAND", "GEWICHTUNGSFAKTOREN", "PFLASTER_PLATTEN",
                                 "GEBRAUCHSWERT", "ZWMSPT"], config, default=.5)
    geb_zwmsph = parse_optional(["POSTPROCESSING", "STRASSENZUSTAND", "GEWICHTUNGSFAKTOREN", "PFLASTER_PLATTEN",
                                 "GEBRAUCHSWERT", "ZWMSPH"], config, default=.25)

    weighting_factors = {"TWGEB": {"ZWSM4L": geb_zwsm4l,
                                   "ZWMSPT": geb_zwmspt,
                                   "ZWMSPH": geb_zwmsph}}

    return weighting_factors


def validate_crs(crs):
    """
    Validates general from of crs. Does NOT check if epsg-code is valid!
    Raises Error if crs is not in compatible form.

    :param str crs: crs to validate.
    :return: crs in form 'epsg:XXXXX'
    :rtype: str
    """
    crs = str(crs)
    if not crs.lower().startswith("epsg:"):
        if not crs.isnumeric():
            raise ValueError(f"EPSG_CODE muss ein gültiger EPSG-Code sein, ist jedoch {crs}!")
        return f"epsg:{crs}"
    return crs.lower()


def validate_gdf_aggregation_areas(path, crs):
    """
    Validates the query area.

    :param str path: path to the query area
    :param str crs: crs
    :return: geodataframe
    :rtype: gpd.GeoDataFrame
    """
    gdf = gpd.read_file(path)

    if gdf.empty:
        raise ValueError('Die Aggregationsflächen enthalten keine Geometrien!')

    if gdf.crs is None:
        gdf = gdf.set_crs(crs)
    else:
        if str(gdf.crs).lower() != crs:
            gdf = gdf.to_crs(crs)

    if not all(isinstance(geometry, Polygon) for geometry in gdf.geometry):
        raise ValueError('Die Aggregationsflächen enthalten nicht ausschließlich Polygon-Geometrien!')

    if not all(geometry.is_valid for geometry in gdf.geometry):
        raise ValueError('Die Aggregationsflächen enthalten ungültige Geometrien.')

    if 'BW' not in gdf.columns:
        raise ValueError("Das Attribut 'BW' (Bauweise) fehlt in den Aggregationsflächen!")

    if 'FK' not in gdf.columns:
        raise ValueError("Das Attribut 'FK' (Funktionsklasse) fehlt in den Aggregationsflächen!")

    if not all(value in ['A', 'P'] for value in gdf['BW']):
        raise ValueError("Werte des Attributs 'FK' (Funktionsklasse) der Aggregationsflächen sind ungültig! "
                         "Gültige Werte: 'A' (Asphalt), 'P' (Pflaster/ Platten)")

    if not all(value in ['A', 'B', 'N'] for value in gdf['FK']):
        raise ValueError("Werte des Attributs 'FK' (Funktionsklasse) der Aggregationsflächen sind ungültig! "
                         "Gültige Werte: 'A' (Hauptverkehrsstraßen), 'B' (Nebenstraßen), 'N' (Nebenflächen)")

    gdf['BW'] = gdf['BW'].astype('category')
    gdf['FK'] = gdf['FK'].astype('category')

    gdf['aggregation_area_id'] = gdf.index

    return gdf


def validate_gdf_pseudo_lanes(path, crs):
    """
    Validates the pseudo lanes.

    :param str path: path to the pseudo lanes
    :param str crs: crs
    :return: geodataframe
    :rtype: gpd.GeoDataFrame
    """
    gdf = gpd.read_file(path)

    if gdf.empty:
        raise ValueError('Die Pseduo-Spuren enthalten keine Geometrien!')

    if gdf.crs is None:
        gdf = gdf.set_crs(crs)
    else:
        if str(gdf.crs).lower() != crs:
            gdf = gdf.to_crs(crs)

    if not all(isinstance(geometry, LineString) for geometry in gdf.geometry):
        raise ValueError('Die Pseduo-Spuren enthalten nicht ausschließlich Linien-Geometrien!')

    if not all(geometry.is_valid for geometry in gdf.geometry):
        raise ValueError('Die Pseduo-Spuren enthalten ungültige Geometrien.')

    return gdf


def validate_parsed_config(parsed_config):
    """
    Validates parsed config-dict.

    :param parsed_config: Parsed config-dict.
    :return: Validated parsed config-dict
    :rtype: dict
    """
    if parsed_config["id_mode"] not in ["Numbered", "Coordinates"]:
        raise ValueError(f"EINGANGSDATEN : PUNKTWOLKEN : ID_TYP muss 'Numbered' oder 'Coordinates' sein aber ist "
                         f"{parsed_config['id_mode']}!")

    if parsed_config["centering"] not in ["Geographic", "Grid", "Recording_Direction", "Recording_Direction_Grid"]:
        raise ValueError(f"EINGANGSDATEN : PANORAMAS : AUSRICHTUNG muss 'Geographic', 'Grid' , 'Recording_Direction' "
                         f"oder 'Recording_Direction_Grid' sein aber ist "
                         f"{parsed_config['centering']}!")

    parsed_config["crs"] = validate_crs(parsed_config["crs"])
    parsed_config["query_area"] = validate_gdf_aggregation_areas(parsed_config["query_area"], parsed_config["crs"])

    if parsed_config["pseudo_lanes"] is not None:
        parsed_config["pseudo_lanes"] = validate_gdf_pseudo_lanes(parsed_config["pseudo_lanes"], parsed_config["crs"])

    return parsed_config


def parse_config(config):
    """
    Parse config for further use.

    :param dict config: Unparsed config
    :return: Parsed config
    :rtype: dict
    """
    parsed_config = {"image_dir": parse_required(["EINGANGSDATEN", "PANORAMAS", "VERZEICHNIS"], config),
                     "image_height": parse_optional(["EINGANGSDATEN", "PANORAMAS", "HOEHE"], config),
                     "image_width": parse_optional(["EINGANGSDATEN", "PANORAMAS", "BREITE"], config),
                     "recording_points": parse_recording_points(config),
                     "centering": parse_optional(["EINGANGSDATEN", "PANORAMAS", "AUSRICHTUNG"], config, default="Geographic"),
                     "car_mask_path": parse_optional(["EINGANGSDATEN", "PANORAMAS", "MASKEN_PFAD"], config),
                     "las_dir": parse_required(["EINGANGSDATEN", "PUNKTWOLKEN", "VERZEICHNIS"], config),
                     "id_mode": parse_optional(["EINGANGSDATEN", "PUNKTWOLKEN", "ID_TYP"], config, default="Numbered"),
                     "id_delimiter": parse_optional(["EINGANGSDATEN", "PUNKTWOLKEN", "ID_TRENNUNG"], config, default="_"),
                     "tile_size": parse_required(["EINGANGSDATEN", "PUNKTWOLKEN", "KACHELGROESSE"], config),
                     "query_area": parse_required(["EINGANGSDATEN", "AGGREGATIONSFLAECHEN"], config),
                     "pseudo_lanes": parse_optional(["EINGANGSDATEN", "PSEUDO_AUFNAHMESPUREN"], config),
                     "recompute": parse_optional(["CACHE_IGNORIEREN"], config, default=False),
                     "out_dir": parse_required(["AUSGABEVERZEICHNIS"], config),
                     "save_raster": parse_optional(["ORTHOS_SPEICHERN"], config, default=True),
                     "crs": parse_optional(["EPSG_CODE"], config, default="epsg:25832"),
                     "simplify": parse_optional(["POSTPROCESSING", "GEOMETRIEN_VEREINFACHEN"], config, default=False),
                     "roadmarkings": {
                         "weighting_factors": parse_weighting_factors(config),
                         "thresholds": parse_thresholds(config)},
                     "roadcondition": {
                         "normalization_factors": {
                             "ASPHALT": parse_roadcondition_normalization_factors_asphalt(config),
                             "PFLASTER_PLATTEN": parse_roadcondition_normalization_factors_pflaster_platten(config)},
                         "weighting_factors": {
                             "ASPHALT": parse_roadcondition_weighting_factors_asphalt(config),
                             "PFLASTER_PLATTEN": parse_roadcondition_weighting_factors_pflaster_platten(config)}}}

    parsed_config["done_file"] = os.path.join(parsed_config["out_dir"], "done.txt")
    parsed_config["tile_pixel_size"] = int(parsed_config["tile_size"] / file_res)
    parsed_config = validate_parsed_config(parsed_config)

    return parsed_config


def create_dir_structure(out_dir, save_raster=False):
    """
    Create the directory structure for all saved outputs.

    :param str out_dir: Output-directory to create.
    :param bool save_raster: Whether to save raster data.
    :return:
    """
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "Geodaten", "Cache", "Markings"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "Geodaten", "Cache", "Substance"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "Geodaten", "Cache", "Planeness"), exist_ok=True)

    if save_raster:
        os.makedirs(os.path.join(out_dir, "Orthos", "RGB"), exist_ok=True)
        os.makedirs(os.path.join(out_dir, "Orthos", "Intensity"), exist_ok=True)
        os.makedirs(os.path.join(out_dir, "Orthos", "Height"), exist_ok=True)


def load_index(index_path):
    """
    Load ids from index file.

    :param index_path: Path to index file.
    :return: ids
    :rtype: list
    """
    if os.path.isfile(index_path):
        with open(index_path, "r") as f:
            ids = f.readlines()
            ids = [line.strip("\n") for line in ids]
    else:
        open(index_path, "a").close()
        ids = []

    return ids


def filter_las_files(las_files, files_to_skip, size, mode, id_delimiter, query_area=None):
    """
    Filter list of filenames by skip-index and query area

    :param list of str las_files: List of filenames of las-files to filter
    :param list of str files_to_skip: List of filenames to skip
    :param gpd.GeoDataFrame query_area: query area to use for spacial selection.
    :param int size: Size of one side of the area in each las-file in meters.
    :param str mode: The way the filenames encode the files position. Options are 'Numbered' oder 'Coordinates'
    :param str id_delimiter: Delimiter used in filename between x and y coordinate
    :return: Filtered Filenames
    :rtype: list of str
    """
    las_files = [filename for filename in las_files
                 if file_id(filename, id_delimiter=id_delimiter) is not None and filename not in files_to_skip]

    if query_area is not None and las_files:
        query_area = query_area.drop(columns=["FileName"], errors="ignore")
        files_gdf = gdf_from_files(las_files, size, mode, id_delimiter, crs=query_area.crs)
        selected_files_gdf = gpd.sjoin(files_gdf, query_area, lsuffix="")
        las_files = list(set(selected_files_gdf["FileName"]))

    return las_files


def load_car_mask(car_mask_path, height=None, width=None):
    """
    Load car-mask. Used to mask out recording car in images.

    :param str car_mask_path: Path to the saved car_mask
    :param int height: Height of the images to mask.
    :param int width: Width of the images to mask.
    :return: Loaded car-mask.
    :rtype: np.ndarray
    """
    car_mask_path = car_mask_path or "data/images/car_mask.png"
    car_mask = cv2.imread(car_mask_path, cv2.IMREAD_COLOR)
    if height is not None and width is not None:
        car_mask = cv2.resize(car_mask, (width, height), interpolation=cv2.INTER_NEAREST)

    return car_mask


def write_to_index(filename, index_file):
    """
    Writes filename to index-file if it is not already in it.

    :param str filename: Name containing a spacial id.
    :param str index_file: Path to index-file to write to.
    :return:
    """

    if filename not in load_index(index_file):
        with open(index_file, "a") as f:
            f.write(filename + "\n")


def corner_coords_from_filename(filename, size, mode, id_delimiter=None):
    """
    Gets the coordinates of the south-west corner of tile represented by the file.

    :param str filename: Name containing a spacial id.
    :param int size: Size of the tile represented by the file.
    :param str mode: The way the filenames encode the files position. Options are 'Numbered' oder 'Coordinates'
    :param str id_delimiter: Delimiter used in filename between x and y coordinate
    :return: Coordinates of the south-west corner of tile represented by the file.
    :rtype: np.ndarray
    """
    if id_delimiter is None:
        id_delimiter = "_"
    coord_multiplier = size if mode == "Numbered" else 1

    corner_coords = np.array(list(map(int, file_id(filename, id_delimiter=id_delimiter).split(id_delimiter)))
                             ) * coord_multiplier

    return corner_coords


def get_bounds(corner_coords, size):
    """
    Gets bounds for tile.

    :param tuple of int or np.ndarray of int or list of int corner_coords: Coordinates of the south-west corner of tile.
    :param int size: Size of the tile.
    :return: Bounds in form (xmin, ymin, xmax, ymax).
    :rtype: tuple of int
    """
    bounds = (corner_coords[0], corner_coords[1], corner_coords[0] + size, corner_coords[1] + size)
    return bounds


def dict_for_tile(filename, corner_coords, size):
    """
    Creates dict for later conversion to GeoDataFrame for tile.

    :param str filename: Filename of the corresponding las-file.
    :param tuple of int or np.ndarray of int or list of int corner_coords: Coordinates of the south-west corner of tile.
    :param int size: Size of the tile.
    :return: Dict containing filename and shapely-polygon for tile.
    :rtype: dict
    """
    points = [corner_coords,
              corner_coords + np.array([0, size]),
              corner_coords + np.array([size, size]),
              corner_coords + np.array([size, 0])]
    polygon = Polygon(points)
    return {"FileName": filename,
            "geometry": polygon}


def gdf_from_files(files, size, mode, id_delimiter, crs):
    """
    Creates a GeoDataFrame of tiles for a list of filenames.

    :param list of str files: List of filenames to create GeoDataFrame for.
    :param int size: Size of each tile.
    :param str mode: The way the filenames encode the files position. Options are 'Numbered' oder 'Coordinates'
    :param str id_delimiter: Delimiter used in filename between x and y coordinate.
    :param crs: Coordinate-reference-system to crate GeoDataFrame in. Must correspond to coordinates encoded in
           filenames.
    :return: GeoDataFrame containing each filenames tile as a Polygon.
    :rtype: gpd.GeoDataFrame
    """
    dicts = [dict_for_tile(filename, corner_coords_from_filename(filename, size, mode, id_delimiter), size)
             for filename in files]

    return gpd.GeoDataFrame(dicts, crs=crs)


def load_images(image_infos, image_dir, image_shape, car_mask=None, centering=None, num_workers=None):
    """
    Loads images based on a list of image infos. Also Masks all Pixels belonging to obstacles.

    :param list of dict image_infos: List of info-dicts for the images to laod.
    :param str image_dir: Base directory to load the images from. Path information in info dicts is relative to this.
    :param tuple of int or list of int  or np.ndarray of int image_shape: Shape of the image in form (height, width).
    :param np.ndarray or None car_mask: (Optional) Array masking the recording car centered in driving direction.
           Must have three channels. A Value of anything but 0 signifies the recording car.
    :param str or None centering: (Optional) Signifies whrere the images to load are centered. Options are
           Geographic (geographic north), Grid (grid north) and Recording_Direction. Default is Geographic.
    :param int or None num_workers: (Optional) Number of parallel Processes to load with.
    :return: Masked-array of shape (num_images_loaded, height, width, 3) containing the loaded and masked images,
             Image infos filtered by correctly loaded images.
    :rtype: (np.ma.masked_array, list of dict)
    """
    if num_workers is None:
        num_workers = os.cpu_count()
    with multiprocessing.Pool(num_workers) as pool:
        loader = Ortho_Creation_Utils.ImageLoader(images_dir=image_dir,
                                                  height=image_shape[0],
                                                  width=image_shape[1],
                                                  centering=centering,
                                                  car_mask=car_mask)
        images = pool.map(loader, image_infos)

    filtered_image_infos = [image_infos[j] for j, image in enumerate(images) if image is not None]
    filtered_images = [image for image in images if image is not None]

    images_array = np.ma.masked_array(filtered_images)

    return images_array, filtered_image_infos


def combine_orthos(rgb_orthos):
    """
    Combines a masked orthos to one ortho containing the best projection available for each pixel.

    :param rgb_orthos: Masked-array of shape (num_orthos, otho_height, ortho_width, 3) representing a list of orthos to
           be combined.
    :return: Combined ortho.
    :rtype: np.ma.masked_array
    """
    rgb_ortho = rgb_orthos[0]  # Choose nearest projection as final one

    if len(rgb_orthos) > 1:  # Filling masked areas only possible if multiple perspectives available

        # Fill masked out areas with next nearest projection
        for k in range(rgb_orthos.shape[0] - 1):
            replacement_indexes = np.all(rgb_ortho.mask == [1, 1, 1], axis=-1)
            rgb_ortho[replacement_indexes] = rgb_orthos[k + 1][replacement_indexes]

    return rgb_ortho


def preprocess_tile(las_filename, tile_bounds, tile_pixel_size, las_dir, image_dir, image_shape,
                    centering, done_file, car_mask, planeness_sampling_points_tile,
                    shapefile_dict=None, id_base=36):
    """
    Create RGB- and intensity-orthos for tile.

    :param str las_filename: Filename of the tiles las-file.
    :param tuple of int tile_bounds: Bounds of the tile in form (xmin, ymin, xmax, ymax).
    :param int tile_pixel_size: Number of pixels in each direction of the orthos to create.
    :param str las_dir: Directory the las-file is in.
    :param str image_dir: Base directory to load the images from. Path information in image-info-shapefile is relative
           to this.
    :param tuple of int image_shape: Shape of the image in form (height, width).
    :param str centering: Signifies whrere the images to load are centered. Options are
           Geographic (geographic north), Grid (grid north) and Recording_Direction. Default is Geographic.
    :param str done_file: Path to index of done tiles.
    :param np.ndarray or None car_mask: (Optional) Array masking the recording car centered in driving direction.
           Must have three channels. A Value of anything but 0 signifies the recording car.
    :param gpd.GeoDataFrame planeness_sampling_points_tile: geodataframe of the planeness sampling points of a tile
    :param shapefile_dict: Dict containing a shapefile and the nessecary keys.
    :param int id_base: base of the id
    :return: List of (rgb_ortho, alpha_rgb, intensity_ortho, alpha_intensity) containing orthos and corresponding
             alpha layers for each height-layer.
    :rtype: list of (tuple of np.ndarray)
    """
    las_filepath = os.path.join(las_dir, las_filename)

    if not os.path.isfile(las_filepath):
        print("No file")
        write_to_index(las_filename, done_file)
        return [], planeness_sampling_points_tile

    tile_full_points = LaserData(las_filepath)  # Load pointcloud as custom LaserData object

    if len(tile_full_points) == 0:
        print("No points")
        write_to_index(las_filename, done_file)
        return [], planeness_sampling_points_tile

    # Get Panorama Metadata for all Panoramas in current area
    tile_full_infos = Ortho_Creation_Utils.get_image_infos_for_area(tile_bounds, shapefile_dict=shapefile_dict)

    if len(tile_full_infos) == 0:
        print("No Infos")
        write_to_index(las_filename, done_file)
        return [], planeness_sampling_points_tile

    # Cluster image Metadata by conflicting height layers (e.g. at Bridges).
    tile_info_clusters = Ortho_Creation_Utils.cluster_recording_infos(tile_full_infos, id_base)

    # Assign points to clusters
    if len(tile_info_clusters) > 1:
        tile_point_clusters = Ortho_Creation_Utils.cluster_laserdata_by_info_clusters(tile_full_points,
                                                                                      tile_info_clusters)
    else:
        tile_point_clusters = [tile_full_points]

    layer_outputs = []

    # Calculate Orthos for each cluster / height layer
    for height_layer, layer_infos in enumerate(tile_info_clusters):
        layer_points = tile_point_clusters[height_layer]  # Select points for layer
        if len(layer_points) == 0:
            continue

        # Classify and filter ground in pointclouds
        ground_mask = Ortho_Creation_Utils.classify_ground(layer_points, tile_bounds, 250)
        filtered_points = layer_points[ground_mask == 1]

        if len(filtered_points) == 0:
            continue

        # Classify and remove outliers in pointclouds
        outlier_mask = Ortho_Creation_Utils.classify_outliers_in_laserdata(filtered_points, radius=0.5, threshold=20)
        filtered_points = filtered_points[outlier_mask != 1]

        if len(filtered_points) == 0:
            continue

        planeness_sampling_points_tile_layer = \
            planeness_sampling_points_tile[planeness_sampling_points_tile["recording_id"].isin(
                [info["image_id"] for info in layer_infos] + ([] if height_layer > 0 else ["_pseudo"]))]

        planeness_heights = sample_points(laserdata=filtered_points,
                                          planeness_sampling_points=planeness_sampling_points_tile_layer)

        planeness_sampling_points_tile.loc[planeness_sampling_points_tile_layer.index, 'z'] = planeness_heights
        planeness_sampling_points_tile.loc[planeness_sampling_points_tile_layer.index, 'height_layer'] = height_layer

        # Create meshgrid for positions to sample point-cloud at
        xx, yy = Ortho_Creation_Utils.sample_meshgrid(tile_bounds, resolution=tile_pixel_size)

        # Create regular grids from filtered point-cloud
        array, mask = Ortho_Creation_Utils.grid_from_points(filtered_points, xx, yy, threshold=1)

        if array is None:
            continue

        intensity_grid = array[..., 1]
        height_grid_unfiltered = array[..., 0]
        height_grid = gaussian_filter(height_grid_unfiltered, sigma=filter_sigma)

        images, layer_infos = load_images(layer_infos, image_dir, image_shape,
                                          car_mask=car_mask, centering=centering)

        if len(layer_infos) == 0:
            continue

        # Construct recording locations
        recording_points = np.array([[image_info["pos"][0], image_info["pos"][1], image_info["height"]]
                                     for image_info in layer_infos])

        # Construct array of positions to sample and array of corresponding recording indexes
        point_mat, image_idx_mat = Ortho_Creation_Utils.create_and_assign_sample_mat(xx, yy,
                                                                                     height_grid,
                                                                                     recording_points,
                                                                                     n_nearest=7)

        if (centering == "Recording_Direction" or centering == "Recording_Direction_Grid") \
                and layer_infos[0]["roll"] is not None and layer_infos[0]["pitch"] is not None:
            recording_angles = np.array([[-image_info["recorderDirection"], -image_info["pitch"], image_info["roll"]]
                                         for image_info in layer_infos])

            rotations = Rotation.from_euler("ZXY", recording_angles, degrees=True)
            rotations = rotations.as_matrix()
            rotation_mat = rotations[image_idx_mat]
        else:
            rotation_mat = None

        # Get grid convergence angle for recording points
        if centering == "Grid" or centering == "Recording_Direction_Grid":
            grid_convergences = np.zeros(len(recording_points))
        else:
            grid_convergences = np.array([Ortho_Creation_Utils.grid_convergence(recording_point)
                                          for recording_point in recording_points])

        recording_point_mat = recording_points[image_idx_mat]  # Construct corresponding array of recording positions

        grid_convergence_mat = grid_convergences[image_idx_mat]  # Construct corresponding array of grid convergences

        # Calculate image coordinates for sampling positions
        pixel_coords = Ortho_Creation_Utils.world_to_image_coordinates(point_mat,
                                                                       recording_point_mat,
                                                                       grid_convergence_mat,
                                                                       images.shape[1], images.shape[2],
                                                                       inverse_rotation=rotation_mat)

        # Combine image index and image coordinates to full index of images-array
        pixel_indexes = np.concatenate([image_idx_mat[..., None], pixel_coords], axis=-1)

        # Project images to orthos
        rgb_orthos = images[tuple(np.moveaxis(pixel_indexes, -1, 0))]
        # Reshape to list of orhtos of length n projected from n-nearest image
        rgb_orthos = np.moveaxis(rgb_orthos, -2, 0)
        rgb_ortho = combine_orthos(rgb_orthos)

        # Transform intensities from 16 bit to 8 bit
        intensity_ortho = (intensity_grid / 2 ** 8).astype(np.uint8)

        height_ortho = ((height_grid_unfiltered - height_grid_unfiltered.min()) / 100 * (2 ** 16 - 1)).astype(np.uint16)

        # Flip y-axis to correct opposing positive image- and world-direction
        rgb_ortho = rgb_ortho[::-1]
        intensity_ortho = intensity_ortho[::-1]
        height_ortho = height_ortho[::-1]
        mask = mask[::-1]

        # Create alpha layer for intensity images
        alpha_intensity = np.zeros((tile_pixel_size, tile_pixel_size), dtype=int)
        alpha_intensity[...] = 255
        alpha_intensity[mask] = 0

        # Create alpha layer for rgb images
        alpha_rgb = np.zeros((tile_pixel_size, tile_pixel_size), dtype=int)
        alpha_rgb[...] = 255
        alpha_rgb[np.all(rgb_ortho.mask == [1, 1, 1], axis=-1)] = 0

        layer_outputs.append((rgb_ortho, alpha_rgb, intensity_ortho, alpha_intensity, height_ortho))

    if not layer_outputs:
        write_to_index(las_filename, done_file)
        return [], planeness_sampling_points_tile

    return layer_outputs, planeness_sampling_points_tile


def predict_for_layer_roadmarkings(rgb_ortho, intensity_ortho):
    """
    Predict roadmarkings for a pair of orthos.

    :param rgb_ortho: 3-channel RGB-ortho.
    :param intensity_ortho: 1-channel intensity-ortho.
    :return: Mask for marking-conditions, Mask for marking-types
    :rtype: (np.ndarray, np.ndarray)
    """
    instance = np.concatenate([rgb_ortho, intensity_ortho[..., np.newaxis]], axis=-1)

    mask = roadmarkings_segmenter.run([],
                                      {roadmarkings_segmenter_input_name: instance[np.newaxis, ...].astype(np.float32)})

    condition_mask = np.squeeze(mask[0].astype(np.uint8))
    type_mask = np.squeeze(mask[1].astype(np.uint8))
    type_mask[type_mask == 6] = 5  # Ignore unused class 6 ("unterbrochener Querstrich")

    return condition_mask, type_mask


def predict_for_layer_substance(rgb_ortho, intensity_ortho, height_ortho):
    """
    Predict roadmarkings for a pair of orthos.

    :param rgb_ortho: 3-channel RGB-ortho.
    :param height_ortho: 1-channel height-ortho.
    :param intensity_ortho: 1-channel intensity-ortho.
    :return: Mask for marking-conditions, Mask for marking-types
    :rtype: (np.ndarray, np.ndarray)
    """
    instance = np.concatenate([rgb_ortho, intensity_ortho[..., np.newaxis]], axis=-1)

    mask = substance_segmenter.run([],
                                   {substance_segmenter_input_name_rgbi: instance[np.newaxis, ...].astype(np.float32),
                                    substance_segmenter_input_name_height: height_ortho[
                                        np.newaxis, ..., np.newaxis].astype(np.float32)})

    mask = np.squeeze(mask[0].astype(np.uint8))

    return mask


def save_as_tiff(path, ortho, tile_corner, tile_size, crs, alpha=None, dtype=np.uint8):
    """
    Saves ortho as geo-tiff.

    :param str path: Path to save to.
    :param np.ndarray ortho: Array of the ortho to save. Must have shape (height, width, 3).
    :param tuple of int or np.ndarray of int or list of int tile_corner: Coordinates of the south-west corner of the
           ortho.
    :param int tile_size: Size in each direction of the tile covered by the ortho.
    :param str crs: Coordinate reference system to save in. All coordinates given must match this.
    :param np.ndarray or None alpha: (Optional) Alpha layer of shape (height, width, 1) for the tiff.
    :param np.dtype dtype: Datatype to use for saving.
    :return:
    """
    tile_pixel_size = int(tile_size / file_res)
    translation = Affine.translation(tile_corner[0], tile_corner[1] + tile_size)
    scaling = Affine.scale(tile_size / tile_pixel_size, -tile_size / tile_pixel_size)
    transform = translation * scaling

    bands = 3 if alpha is None else 4

    with rasterio.open(
            path,
            'w',
            driver='GTiff',
            height=tile_pixel_size,
            width=tile_pixel_size,
            count=bands,
            dtype=dtype,
            crs=crs,
            transform=transform,
    ) as dst:
        dst.write(ortho[..., 0], 1)
        dst.write(ortho[..., 1], 2)
        dst.write(ortho[..., 2], 3)
        if alpha is not None:
            dst.write(alpha, 4)


def save_orthos(rgb_ortho, alpha_rgb, intensity_ortho, alpha_intensity_height, height_ortho, tile_corner, tile_id,
                height_layer, tile_size, out_dir, crs):
    """
    Saves RGB- and intensity-orthos as geo-tiffs.

    :param np.ndarray rgb_ortho: RGB-ortho array of shape (height, width, 3).
    :param np.ndarray alpha_rgb: Alpha-layer for RGB-ortho.
    :param np.ndarray intensity_ortho: Intensity-ortho array of shape (height, width, 1).
    :param np.ndarray height_ortho: Height-ortho array of shape (height, width, 1).
    :param np.ndarray alpha_intensity_height: Alpha-layer for intensity-ortho and height-ortho.
    :param tuple of int or np.ndarray of int or list of int tile_corner:  Coordinates of the south-west corner of the
           ortho.
    :param str tile_id: Id of the tile the orthos cover.
    :param int height_layer: Height-layer the orthos are in.
    :param int tile_size: Size in each direction of the tile covered by the orthos.
    :param str out_dir: Path to base output directory.
    :param str crs: Coordinate reference system to save in. All coordinates given must match this.
    :return:
    """
    rgb_path = os.path.join(out_dir, "Orthos", "RGB", f"{tile_id}_h{height_layer}_rgb.tiff")
    save_as_tiff(rgb_path, rgb_ortho, tile_corner, tile_size, crs=crs,
                 alpha=alpha_rgb)

    intensity_path = os.path.join(out_dir, "Orthos", "Intensity", f"{tile_id}_h{height_layer}_intensity.tiff")
    intensity_ortho_3_channel = np.repeat(intensity_ortho[..., np.newaxis], 3, axis=-1)
    save_as_tiff(intensity_path, intensity_ortho_3_channel, tile_corner, tile_size, crs=crs,
                 alpha=alpha_intensity_height)

    height_path = os.path.join(out_dir, "Orthos", "Height", f"{tile_id}_h{height_layer}_height.tiff")
    height_ortho_3_channel = np.repeat(height_ortho[..., np.newaxis], 3, axis=-1)
    # noinspection PyTypeChecker
    save_as_tiff(height_path, height_ortho_3_channel, tile_corner, tile_size, crs=crs,
                 alpha=alpha_intensity_height, dtype=np.uint16)


def process_tile(las_filename, parsed_config, car_mask, planeness_sampling_points):
    """
    Process single tile. Computes Orthos, predicts masks, postprocesses masks and saves results.

    :param str las_filename: Filename of the tiels las-file.
    :param dict parsed_config: Dictionary containing parsed and validated info from config file.
    :param np.ndarray or None car_mask: (Optional) Array masking the recording car centered in driving direction.
           Must have three channels. A Value of anything but 0 signifies the recording car.
    :param gpd.GeoDataFrame planeness_sampling_points: geodataframe of the planeness sampling points
    :return:
    """
    id_delimiter = parsed_config["id_delimiter"]
    id_base = parsed_config["recording_points"]["id_base"]
    tile_size = parsed_config["tile_size"]
    mode = parsed_config["id_mode"]
    las_dir = parsed_config["las_dir"]
    image_dir = parsed_config["image_dir"]
    image_height = parsed_config["image_height"]
    image_width = parsed_config["image_width"]
    centering = parsed_config["centering"]
    out_dir = parsed_config["out_dir"]
    save_raster = parsed_config["save_raster"]
    # factors = parsed_config["roadmarkings"]["weighting_factors"]
    thresholds = parsed_config["roadmarkings"]["thresholds"]
    tile_pixel_size = parsed_config["tile_pixel_size"]
    done_file = parsed_config["done_file"]
    shapefile_dict = parsed_config["recording_points"]
    crs = parsed_config["crs"]

    tile_id = file_id(las_filename, id_delimiter=id_delimiter)  # Identifier of current area
    tile_corner = corner_coords_from_filename(las_filename, tile_size, mode, id_delimiter)
    # (xmin, ymin, xmax, ymax) for current area
    tile_bounds = get_bounds(tile_corner, tile_size)

    planeness_sampling_points_tile = planeness_sampling_points.cx[tile_bounds[0]:tile_bounds[2],
                                                                  tile_bounds[1]:tile_bounds[3]]

    ortho_layers, planeness_sampling_points_tile = preprocess_tile(las_filename, tile_bounds, tile_pixel_size, las_dir,
                                                                   image_dir, (image_height, image_width), centering,
                                                                   done_file, car_mask, planeness_sampling_points_tile,
                                                                   shapefile_dict=shapefile_dict, id_base=id_base)

    for height_layer, layer_outputs in enumerate(ortho_layers):
        rgb_ortho, alpha_rgb, intensity_ortho, alpha_intensity, height_ortho = layer_outputs

        condition_mask, type_mask = predict_for_layer_roadmarkings(subsample_ortho(rgb_ortho, factor=2),
                                                                   subsample_ortho(intensity_ortho, factor=2))

        substance_mask = predict_for_layer_substance(rgb_ortho, intensity_ortho, height_ortho)

        if save_raster:
            save_orthos(rgb_ortho, alpha_rgb, intensity_ortho, alpha_intensity, height_ortho, tile_corner, tile_id,
                        height_layer, tile_size, out_dir, crs=crs)

        roadmarkings_gdf = Postprocessing_Utils.get_roadmarkings_gdf(condition_mask, type_mask, tile_corner,
                                                                     file_res * 2, thresholds,
                                                                     height_layer=height_layer, crs=crs)

        substances_gdf = Postprocessing_Utils.get_substances_gdf(substance_mask, tile_corner, file_res,
                                                                 height_layer=height_layer, crs=crs)

        if len(roadmarkings_gdf) > 0:
            roadmarkings_gdf.to_feather(
                os.path.join(out_dir, "Geodaten", "Cache", "Markings",
                             f"{tile_id}_h{height_layer}_markings.feather"))

        if len(substances_gdf) > 0:
            substances_gdf.to_feather(
                os.path.join(out_dir, "Geodaten", "Cache", "Substance",
                             f"{tile_id}_h{height_layer}_substance.feather"))

    if len(planeness_sampling_points_tile) > 0:
        planeness_sampling_points.update(planeness_sampling_points_tile)
        planeness_sampling_points.to_feather(os.path.join(out_dir, "Geodaten", "Cache", "Planeness",
                                                          "planeness.feather"))

    write_to_index(las_filename, done_file)


def process_files(las_files, parsed_config):
    """
    Processes a list of tiles based on filenames of corresponding las-files.

    :param list of str las_files: List of las-filenames to process.
    :param dict parsed_config: Dictionary containing parsed and validated info from config file.
    :return:
    """
    total_files = len(las_files)

    if not total_files:
        print("Keine neuen zu verarbeitenden Dateien im ausgewählten Gebiet gefunden!")
        return

    times = []

    car_mask = load_car_mask(parsed_config["car_mask_path"],
                             parsed_config["image_height"],
                             parsed_config["image_width"])

    if parsed_config["recording_points"] is not None:
        point_query_area = gdf_from_files(las_files, parsed_config["tile_size"], parsed_config["id_mode"],
                                          parsed_config["id_delimiter"], crs=parsed_config["crs"])
        point_query_area['geometry'] = point_query_area.buffer(tile_buffer)
        shapefile = gpd.read_file(parsed_config["recording_points"]["path"], bbox=point_query_area)
        shapefile.to_crs(parsed_config["crs"], inplace=True)
        query_shapefile = gpd.clip(shapefile, point_query_area)
        query_shapefile = query_shapefile[shapefile.columns].reset_index(drop=True)
        parsed_config["recording_points"]["file"] = query_shapefile

    planeness_sampling_points = create_planeness_sampling_points(gdf=parsed_config["recording_points"]["file"],
                                                                 id_column=parsed_config["recording_points"]["id_key"],
                                                                 id_base=parsed_config["recording_points"]["id_base"],
                                                                 gdf_pseudo_lanes=parsed_config["pseudo_lanes"],
                                                                 crs=parsed_config["crs"])

    for i, las_file in enumerate(las_files):
        print(f"Verarbeite Kachel Nr. {i + 1} von {total_files} mit ID "
              f"{file_id(las_file, parsed_config['id_delimiter'])}.")
        start = time.perf_counter()
        process_tile(las_file, parsed_config, car_mask, planeness_sampling_points)
        times.append(time.perf_counter() - start)
        print(f"Kachel verarbeitet in {time.perf_counter() - start} s.")

    print(f"Verarbeitung von {total_files} Kacheln abgeschlossen. Durchschnittliche Zeit pro Kachel: "
          f"{float('%.2f' % float(sum(times) / len(times)))} s.")


def join_gdfs(shapes_dir):
    """
    Combines all shapefiles in a directory.

    :param str shapes_dir: Directory the shapefiles to combine are in.
    :return: joined geodataframe
    :rtype: gpd.GeoDataFrame
    """
    files = os.listdir(shapes_dir)
    shapes = []
    for file in files:
        if file.endswith(".feather"):
            shape = gpd.read_feather(os.path.join(shapes_dir, file))
            shapes.append(shape)
    joined_shapes = gpd.GeoDataFrame(pandas.concat(shapes, ignore_index=True), crs=shapes[0].crs)

    return joined_shapes


def reweigh_tiles(shape_path, factors, thresholds):
    """
    Reweighs a shapefile of tiles.

    :param str shape_path: Path to the shapefile to reweigh.
    :param np.ndarray factors: Weighting factors.
    :param dict thresholds: Dict containing classification thresholds.
    :return:
    """
    tiles_shape = gpd.read_file(shape_path)
    tiles_shape = Postprocessing_Utils.classify_tile_dataframe(tiles_shape, factors, thresholds)
    tiles_shape.to_file(shape_path, schema=Postprocessing_Utils.tile_schema)


def reweigh_markings(shape_path, thresholds):
    """
    Reweighs a shapefile of markings.

    :param str shape_path: Path to the shapefile to reweigh.
    :param dict thresholds: Dict containing classification thresholds.
    :return:
    """
    markings_shape = gpd.read_file(shape_path)
    markings_shape = Postprocessing_Utils.classify_marking_dataframe(markings_shape, thresholds)
    markings_shape.to_file(shape_path, schema=Postprocessing_Utils.object_schema)


def simplify_gdf(gdf, tolerance=file_res):
    """
    Reduces number of points in polygons of a geodataframe.

    :param gpd.GeoDataFrame gdf: Geodataframe
    :param float tolerance: Tolerance to use in Douglas-Peucker algorithm
    :return: simplified geodataframe
    :rtype: gpd.GeoDataFrame
    """
    gdf.geometry = gdf.simplify(tolerance)
    return gdf
