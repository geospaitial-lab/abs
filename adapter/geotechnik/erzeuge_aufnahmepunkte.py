# @author: Roß, Alexander - Fachbereich Elektrotechnik, Westfälische Hochschule Gelsenkirchen

import argparse
import json
import math
import os
from datetime import datetime

import geopandas as gpd
import pandas
from pytz import timezone


def make_recording_points(args_):
    meta_path = args_.meta_path
    save_meta_path = args_.out_path

    print("Konvertiere Aufnahmepunkte und Punktwolken Abdeckung.")

    os.makedirs(save_meta_path, exist_ok=True)

    recordings_save_path = os.path.join(save_meta_path, f"recordings")

    last_id = 0

    recordings_shapes = []

    sub_dirs = [sub_dir for sub_dir in os.listdir(meta_path) if not sub_dir.startswith(".")]
    for sub_dir in sub_dirs:
        for sub_sub_dir in os.listdir(os.path.join(meta_path, sub_dir)):
            if sub_sub_dir.startswith("."):
                continue
            with open(os.path.join(meta_path, sub_dir, sub_sub_dir, "meta.json")) as file:
                meta_dir = json.load(file)

            recordings_dict = [{"sensor_id": image["sensor_id"],
                                "image_path": os.path.join(sub_dir, sub_sub_dir, image["path"]),
                                "time_stamp": datetime.fromtimestamp(image["time_stamp"],
                                                                     tz=timezone("Europe/Berlin")).isoformat(),
                                "x": image["pose"]["translation"][0],
                                "y": image["pose"]["translation"][1],
                                "height": image["pose"]["translation"][2],
                                "roll": image["pose"]["orientation_roll_pitch_yaw"][0] * 180 / math.pi,
                                "pitch": -(image["pose"]["orientation_roll_pitch_yaw"][1] * 180 / math.pi + 90),
                                "yaw": image["pose"]["orientation_roll_pitch_yaw"][2] * 180 / math.pi,
                                "image_id": str(i + last_id + 1),
                                "run_id": sub_dir
                                } for i, image in enumerate(meta_dir["spherical_images"])
                               if image["sensor_id"] == "ladybug_front"]

            last_id = int(recordings_dict[-1]["image_id"])

            recordings_shape = gpd.GeoDataFrame(recordings_dict,
                                                geometry=gpd.points_from_xy([r_dict["x"] for r_dict in recordings_dict],
                                                                            [r_dict["y"] for r_dict in
                                                                             recordings_dict]),
                                                crs="EPSG:25832")
            recordings_shapes.append(recordings_shape)

    joint_recordings_shapes = gpd.GeoDataFrame(pandas.concat(recordings_shapes, ignore_index=True),
                                               crs=recordings_shapes[0].crs)

    joint_recordings_shapes.to_file(recordings_save_path, driver="GPKG")

    print("Verarbeitung abgeschlossen.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="abs-geotechnik-aufnahmepunkte")
    parser.add_argument("meta_path", help="Hauptverzeichnis mit meta.json Dateien"
                                          " in entsprechenden Unterverzeichnissen")
    parser.add_argument("out_path", help="Ausgabeverzeichnis")

    args = parser.parse_args()

    make_recording_points(args)
