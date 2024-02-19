# @author: Roß, Alexander - Fachbereich Elektrotechnik, Westfälische Hochschule Gelsenkirchen
# @coauthor: Maryniak, Marius - Fachbereich Elektrotechnik, Westfälische Hochschule Gelsenkirchen

import argparse
import os
import time
import warnings

import geopandas as gpd
import yaml
from rasterio.errors import ShapeSkipWarning

from src import General_Utils
from src.Evaluator import evaluate
from src.logo import logo
from src.Planeness_Dataframe_Handler import process_gdf_planeness
from src.Substance_Dataframe_Handler import process_gdf_substance


def main(args_):
    print(logo)

    print("Starte Verarbeitung.")
    with open(args_.config) as config_file:
        config = yaml.safe_load(config_file)

    parsed_config = General_Utils.parse_config(config)

    General_Utils.create_dir_structure(parsed_config["out_dir"], save_raster=parsed_config["save_raster"])

    done_files = General_Utils.load_index(parsed_config["done_file"])
    files_to_skip = [] if parsed_config["recompute"] else done_files

    las_files = os.listdir(parsed_config["las_dir"])
    las_files = General_Utils.filter_las_files(las_files, files_to_skip,
                                               parsed_config["tile_size"], parsed_config["id_mode"],
                                               parsed_config["id_delimiter"], query_area=parsed_config["query_area"])

    General_Utils.process_files(las_files, parsed_config)

    print("Starte Nachverarbeitung.")

    gdf_markings = General_Utils.join_gdfs(os.path.join(parsed_config["out_dir"], "Geodaten", "Cache", "Markings"))
    gdf_heights = gpd.read_feather(os.path.join(parsed_config["out_dir"], "Geodaten", "Cache", "Planeness",
                                                "planeness.feather"))
    gdf_substance = General_Utils.join_gdfs(os.path.join(parsed_config["out_dir"], "Geodaten", "Cache", "Substance"))

    if parsed_config["simplify"]:
        gdf_markings = General_Utils.simplify_gdf(gdf_markings)
        gdf_substance = General_Utils.simplify_gdf(gdf_substance)

    gdf_markings.to_file(os.path.join(parsed_config["out_dir"], "Geodaten", "Strassenmarkierungen.gpkg"), driver="GPKG")

    gdf_substance.to_file(os.path.join(parsed_config["out_dir"], "Geodaten", "Substanz.gpkg"), driver="GPKG")

    gdf_aggregation_areas = parsed_config["query_area"]

    print("Starte Ebenheitsberechnung.")
    start_time = time.perf_counter()
    gdf_planeness = process_gdf_planeness(gdf_planeness=gdf_heights,
                                          gdf_aggregation_areas=gdf_aggregation_areas,
                                          crs=parsed_config["crs"])
    print(f"Ebenheitsberechnung abgeschlossen in {time.perf_counter() - start_time} s.")

    print("Starte Aggregierung.")
    start_time = time.perf_counter()
    gdf_aggregation_areas_asphalt = gdf_aggregation_areas[gdf_aggregation_areas["BW"] == "A"]
    gdf_substance_asphalt = process_gdf_substance(gdf_aggregation_areas=gdf_aggregation_areas_asphalt,
                                                  gdf_substance=gdf_substance,
                                                  crs=parsed_config["crs"],
                                                  mode="asphalt")

    gdf_aggregation_areas_pflaster_platten = gdf_aggregation_areas[gdf_aggregation_areas["BW"] == "P"]
    gdf_substance_pflaster_platten = process_gdf_substance(gdf_aggregation_areas=gdf_aggregation_areas_pflaster_platten,
                                                           gdf_substance=gdf_substance,
                                                           crs=parsed_config["crs"],
                                                           mode="pflaster_platten")
    print(f"Aggregierung abgeschlossen in {time.perf_counter() - start_time} s.")

    print("Starte Evaluierung.")
    start_time = time.perf_counter()
    gdf_evaluated = evaluate(gdf_planeness=gdf_planeness,
                             gdf_substance_asphalt=gdf_substance_asphalt,
                             gdf_substance_pflaster_platten=gdf_substance_pflaster_platten,
                             normalization_factors=parsed_config["roadcondition"]["normalization_factors"],
                             weighting_factors=parsed_config["roadcondition"]["weighting_factors"])
    print(f"Evaluierung abgeschlossen in {time.perf_counter() - start_time} s.")

    gdf_evaluated['BW'] = gdf_evaluated['BW'].astype(str)
    gdf_evaluated['FK'] = gdf_evaluated['FK'].astype(str)

    gdf_evaluated.to_file(os.path.join(parsed_config["out_dir"], "Geodaten", "Strassenzustand.gpkg"), driver="GPKG")

    print("Verarbeitung abgeschlossen.")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=ShapeSkipWarning)

    parser = argparse.ArgumentParser(description="abs")
    parser.add_argument("config", help="Konfigurationsdatei")

    args = parser.parse_args()

    main(args)
