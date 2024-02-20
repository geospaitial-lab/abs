# @author: Roß, Alexander - Fachbereich Elektrotechnik, Westfälische Hochschule Gelsenkirchen

import argparse
import json
import os
import warnings

import geopandas as gpd
import numpy as np
import pandas
from shapely.geometry import box

from src.GSC_Utils import LaserData, concatenate_laserdata

tile_size = 50  # in m


def retile_pointclouds(args_):
    meta_path = args_.meta_path
    las_path = args_.las_path
    query_path = args_.query_path
    out_path = args_.out_path

    os.makedirs(os.path.join(out_path), exist_ok=True)

    print("Berechne Punktwolkenabdeckung.")

    pointcloud_shapes = []
    sub_dirs = [sub_dir for sub_dir in os.listdir(meta_path) if not sub_dir.startswith(".")]
    for sub_dir in sub_dirs:
        for sub_sub_dir in os.listdir(os.path.join(meta_path, sub_dir)):
            if sub_sub_dir.startswith("."):
                continue
            with open(os.path.join(meta_path, sub_dir, sub_sub_dir, "meta.json")) as file:
                meta_dir = json.load(file)

            pointclouds_dict = [{"sensor_id": item["sensor_id"],
                                 "path": os.path.join(sub_dir, sub_sub_dir, item["path"]),
                                 "geometry": box(item["bounding_box"]["min"][0],
                                                 item["bounding_box"]["min"][1],
                                                 item["bounding_box"]["max"][0],
                                                 item["bounding_box"]["max"][1])}
                                for item in meta_dir["point_clouds"] if item["sensor_id"] == "CPS"]
            pointcloud_shape = gpd.GeoDataFrame(pointclouds_dict, crs="EPSG:25832")
            pointcloud_shapes.append(pointcloud_shape)

    las_shape = gpd.GeoDataFrame(pandas.concat(pointcloud_shapes, ignore_index=True),
                                 crs=pointcloud_shapes[0].crs)

    query_shape = gpd.read_file(query_path)

    las_shape = gpd.sjoin(las_shape, query_shape)

    if len(las_shape) == 0:
        warnings.warn("Keine Punktwolken in zu berechnendem Bereich gefunden!")
        return

    las_bounds = (las_shape.total_bounds // tile_size + np.array([0, 0, 1, 1])) * tile_size

    num_tiles = (las_bounds / tile_size)[2:] - (las_bounds / tile_size)[:2]

    done_pointclouds = os.listdir(os.path.join(out_path))

    print("Speichere neue Punktwolken.")

    for id_x in range(int(num_tiles[0])):
        for id_y in range(int(num_tiles[1])):

            print(f"Verarbeite Kachel {id_x + 1}/{int(num_tiles[0])} | {id_y + 1}/{int(num_tiles[1])}")

            coord_x = int((las_bounds[0] + id_x * tile_size) // tile_size)
            coord_y = int((las_bounds[1] + id_y * tile_size) // tile_size)

            if f"CPS_{coord_x}_{coord_y}.laz" in done_pointclouds:
                continue
            tile_shapes = las_shape.cx[coord_x * tile_size:(coord_x + 1) * tile_size,
                          coord_y * tile_size:(coord_y + 1) * tile_size]

            pointclouds = []
            for path in set(tile_shapes["path"]):
                pointcloud_path = os.path.join(las_path, path)
                pointcloud = LaserData(pointcloud_path)
                pointclouds.append(pointcloud)

            if len(pointclouds) <= 0:
                continue

            full_pointcloud = concatenate_laserdata(pointclouds)

            tile_pointcloud = full_pointcloud.cx[coord_x * tile_size:(coord_x + 1) * tile_size,
                              coord_y * tile_size:(coord_y + 1) * tile_size]
            if len(tile_pointcloud) > 0:
                tile_pointcloud.save(os.path.join(out_path, f"CPS_{coord_x}_{coord_y}.laz"))

    print("Verarbeitung abgeschlossen.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="abs-geotechnik-punktwolken")
    parser.add_argument("meta_path", help="Hauptverzeichnis mit meta.json Dateien"
                                          " in entsprechenden Unterverzeichnissen")
    parser.add_argument("las_path", help="Hauptverzeichnis mit .las- oder .laz-Dateien Dateien"
                                         " in entsprechenden Unterverzeichnissen")
    parser.add_argument("query_path", help=".shp- oder .gpkg-Datei mit Polygonen, welche den zu verarbeitenden Bereich"
                                           "abdecken,z.B. Aggregationsflächen.")
    parser.add_argument("out_path", help="Ausgabeverzeichnis für die Punktwolken")

    args = parser.parse_args()

    retile_pointclouds(args)
