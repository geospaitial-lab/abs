# @author: Roß, Alexander - Fachbereich Elektrotechnik, Westfälische Hochschule Gelsenkirchen
# @coauthor: Maryniak, Marius - Fachbereich Elektrotechnik, Westfälische Hochschule Gelsenkirchen

import os
import time

import argparse
import geopandas as gpd
import yaml

from src import General_Utils
from src.Evaluator import reevaluate
from src.logo import logo


def main(args_):
    print(logo)

    print("Starte Neugewichtung.")
    start_time = time.perf_counter()

    with open(args_.config) as config_file:
        config = yaml.safe_load(config_file)

    parsed_config = General_Utils.parse_config(config)

    gdf = gpd.read_file(os.path.join(parsed_config["out_dir"], "Geodaten", "Strassenzustand.gpkg"))

    gdf_evaluated = reevaluate(gdf=gdf,
                               normalization_factors=parsed_config["roadcondition"]["normalization_factors"],
                               weighting_factors=parsed_config["roadcondition"]["weighting_factors"])

    if args.out_path is not None:
        gdf_evaluated.to_file(args.out_path, driver="GPKG")
    else:
        gdf_evaluated.to_file(os.path.join(parsed_config["out_dir"], "Geodaten", "Strassenzustand.gpkg"), driver="GPKG")

    print(f"Evaluierung abgeschlossen in {time.perf_counter() - start_time} s.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="abs - Neugewichtung")
    parser.add_argument("config", help="Konfigurationsdatei")
    parser.add_argument("out_path", help="Pfad zur Ausgabedatei", default=None)

    args = parser.parse_args()

    main(args)
