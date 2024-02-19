# @author: Roß, Alexander - Fachbereich Elektrotechnik, Westfälische Hochschule Gelsenkirchen

import argparse
import os
import warnings

import geopandas as gpd
import yaml

from src.Evaluator import evaluate
from src.General_Utils import (
    parse_required,
    validate_gdf_aggregation_areas,
    parse_roadcondition_normalization_factors_asphalt,
    parse_roadcondition_normalization_factors_pflaster_platten,
    parse_roadcondition_weighting_factors_asphalt,
    parse_roadcondition_weighting_factors_pflaster_platten,
)

from src.Planeness_Dataframe_Handler import process_gdf_planeness
from src.Substance_Dataframe_Handler import process_gdf_substance


def evaluate_only(args_):
    with open(args_.config) as config_file:
        config = yaml.safe_load(config_file)

    out_dir = parse_required(["AUSGABEVERZEICHNIS"], config)
    crs = parse_required(["EPSG_CODE"], config)
    gdf_aggregation_areas = validate_gdf_aggregation_areas(parse_required(["EINGANGSDATEN", "AGGREGATIONSFLAECHEN"],
                                                                          config), crs=crs)

    normalization_factors = {"ASPHALT": parse_roadcondition_normalization_factors_asphalt(config),
                             "PFLASTER_PLATTEN": parse_roadcondition_normalization_factors_pflaster_platten(config)}
    weighting_factors = {"ASPHALT": parse_roadcondition_weighting_factors_asphalt(config),
                         "PFLASTER_PLATTEN": parse_roadcondition_weighting_factors_pflaster_platten(config)}

    # gdf_markings = gpd.read_file(os.path.join(out_dir, "Shapes", "combined_markings.gpkg"))
    gdf_heights = gpd.read_feather(os.path.join(out_dir, "Shapes", "Planeness", "planeness.feather"))
    gdf_substance = gpd.read_file(os.path.join(out_dir, "Shapes", "combined_substance.gpkg"))

    gdf_planeness = process_gdf_planeness(gdf_planeness=gdf_heights,
                                          gdf_aggregation_areas=gdf_aggregation_areas,
                                          crs=crs)

    gdf_aggregation_areas_asphalt = gdf_aggregation_areas[gdf_aggregation_areas["BW"] == "A"]
    gdf_substance_asphalt = process_gdf_substance(gdf_aggregation_areas=gdf_aggregation_areas_asphalt,
                                                  gdf_substance=gdf_substance,
                                                  crs=crs,
                                                  mode="asphalt")

    gdf_aggregation_areas_pflaster_platten = gdf_aggregation_areas[gdf_aggregation_areas["BW"] == "P"]
    gdf_substance_pflaster_platten = process_gdf_substance(gdf_aggregation_areas=gdf_aggregation_areas_pflaster_platten,
                                                           gdf_substance=gdf_substance,
                                                           crs=crs,
                                                           mode="pflaster_platten")

    gdf_evaluated = evaluate(gdf_planeness=gdf_planeness,
                             gdf_substance_asphalt=gdf_substance_asphalt,
                             gdf_substance_pflaster_platten=gdf_substance_pflaster_platten,
                             normalization_factors=normalization_factors,
                             weighting_factors=weighting_factors)

    gdf_evaluated['BW'] = gdf_evaluated['BW'].astype(str)
    gdf_evaluated['FK'] = gdf_evaluated['FK'].astype(str)

    gdf_evaluated.to_file(os.path.join(out_dir, "Shapes", "evaluated.gpkg"), driver="GPKG")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)

    parser = argparse.ArgumentParser(
        description="Neugewichten der Shapefiles.")
    parser.add_argument("config", help="YAML-Datei mit Konfiguration der Postprocessing Parameter.")

    args = parser.parse_args()

    evaluate_only(args)
