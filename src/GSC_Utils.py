# @author: Roß, Alexander - Fachbereich Elektrotechnik, Westfälische Hochschule Gelsenkirchen
# @coauthor: Kuhlmann, Christian - Fachbereich Elektrotechnik, Westfälische Hochschule Gelsenkirchen

import os
import re

import laspy
import numpy as np


class LasException(Exception):
    """
    Exception for LaserData class
    """
    pass


class LaserData:
    """
    Basic datastructure for laser data. Minimal Version.

    Author: Christian Kuhlmann + Alexander Roß
    """

    def __init__(self, path=None):
        """
        constructor, can optionally provide a laz-file to be imported
        :param path: path to a laz-file or las-file
        """
        self.data = laspy.LasData(laspy.header.LasHeader())
        if path:
            self.load_laz(path)

    @classmethod
    def from_lasdata(cls, lasdata):
        """
        alternative constructor to load from laspy.LasData object

        :param lasdata: laspy.LasData object
        :return:
        """
        ld = cls()
        ld.data = lasdata

        return ld

    @classmethod
    def from_numpy(cls, array):
        """
        alternative constructor to load from numpy arrays

        :param np.ndarray array: numpy structured array or plain numpy array of shape (len(points), 3)
        :return:

        structured arrays need to be of dtype =
         [('x', '<f8'),
          ('y', '<f8'),
           ('z', '<f8'),
            ('intensity', '<u2'),
             ('return_number', 'u1'),
              ('number_of_returns', 'u1'),
               ('scan_direction_flag', 'u1'),
                ('edge_of_flight_line', 'u1'),
                 ('classification', 'u1'),
                  ('scan_angle_rank', '<f4'),
                   ('user_data', 'u1'),
                    ('point_source_id', '<u2'),
                     ('gps_time', '<f8'),
                      ('red', '<u2'),
                       ('green', '<u2'),
                        ('blue', '<u2')]
        except for 'x', 'y' and 'z' all fields are optional.

        """
        if len(array) == 0:
            return cls()
        structured = array.dtype.names is not None

        header = laspy.LasHeader(point_format=3, version="1.2")
        if structured:
            header.offsets = np.array([array["x"].min(), array["y"].min(), array["z"].min()])
        else:
            header.offsets = np.min(array, axis=0)
        header.scales = np.array([0.001, 0.001, 0.001])
        header.global_encoding.gps_time_type = laspy.header.GpsTimeType.STANDARD

        lasdata = laspy.LasData(header)

        if structured:
            lasdata.x = array["x"]
            lasdata.y = array["y"]
            lasdata.z = array["z"]

            for key in array.dtype.fields:
                if key not in ["x", "y", "z"]:
                    lasdata[key] = array[key]
        else:
            lasdata.x = array[:, 0]
            lasdata.y = array[:, 1]
            lasdata.z = array[:, 2]

        return cls.from_lasdata(lasdata)

    def load_laz(self, path):
        """
        loads laz-file or las-file into self.data
        :param path: path to a laz-file or las-file
        :return: None
        """
        if path and os.path.isfile(path):
            with laspy.open(path) as fh:
                self.data = fh.read()
        else:
            raise LasException('No file ' + str(path))

    def numpy(self):
        """
        Returns numpy structured array of dtype=[('x', '<f8'), ('y', '<f8'), ('z', '<f8'), ('intensity', '<u2'),
                                                       ('return_number', 'u1'), ('number_of_returns', 'u1'),
                                                       ('scan_direction_flag', 'u1'), ('edge_of_flight_line', 'u1'),
                                                       ('classification', 'u1'), ('scan_angle_rank', '<f4'),
                                                       ('user_data', 'u1'), ('point_source_id', '<u2'),
                                                       ('gps_time', '<f8'),
                                                       ('red', '<u2'), ('green', '<u2'), ('blue', '<u2')]
        :return: numpy structured array
        """
        if not self.data:
            return None

        numpy_struct = np.zeros(len(self.data), dtype=[('x', '<f8'), ('y', '<f8'), ('z', '<f8'), ('intensity', '<u2'),
                                                       ('return_number', 'u1'), ('number_of_returns', 'u1'),
                                                       ('scan_direction_flag', 'u1'), ('edge_of_flight_line', 'u1'),
                                                       ('classification', 'u1'), ('scan_angle_rank', '<f4'),
                                                       ('user_data', 'u1'), ('point_source_id', '<u2'),
                                                       ('gps_time', '<f8'),
                                                       ('red', '<u2'), ('green', '<u2'), ('blue', '<u2')])

        numpy_struct["x"] = self.data.x
        numpy_struct["y"] = self.data.y
        numpy_struct["z"] = self.data.z

        for key in numpy_struct.dtype.fields:
            if key not in ["x", "y", "z"] and key in self.data.point_format.dimension_names:
                numpy_struct[key] = self.data[key]

        return numpy_struct

    @property
    def bounds(self):
        """
        Returns the min and max bounds of the pointcloud.
        :return: Tuple with bounds: (xmin, ymin, zmin, xmax, ymax, zmax).
        :rtype: tuple
        """
        xmin, xmax = self.x.min(), self.x.max()
        ymin, ymax = self.y.min(), self.y.max()
        zmin, zmax = self.z.min(), self.z.max()

        return xmin, ymin, zmin, xmax, ymax, zmax

    @property
    def cx(self):
        """
        Coordinate based indexing similar to geopandas.
        :return: Indexer object that implements coordinate based indexing in __getitem__.
        """
        return _CoordinateIndexer(self)

    def save(self, path):
        """
        Saves pointcloud
        :param path: Path to save to.
        """
        self.data.points.array = np.ascontiguousarray(self.data.points.array)
        self.data.write(path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        if type(item) == str:
            return self.data.__getitem__(item)
        if type(item) == int:
            item = slice(item, item + 1)
        result = laspy.LasData(self.data.header)
        result.points = self.data.points[item]
        return LaserData.from_lasdata(lasdata=result)

    def __setitem__(self, key, value):
        if type(key) == str:
            if type(value) == int:
                value = [value]
            self.data.__setitem__(key, value)
        else:
            raise TypeError("Item assignment only implemented for str keys!")

    def __getattr__(self, item):
        try:
            return self.data.__getattribute__(item)
        except AttributeError:
            return self.data.__getattr__(item)

    def __setattr__(self, key, value):
        if key != "data":
            self.data.__setattr__(key, value)
        else:
            super(LaserData, self).__setattr__(key, value)

    def __iter__(self):
        return iter(self.data)


class _CoordinateIndexer:
    """
    Implements coordinate based indexing similar to geopandas.
    """
    def __init__(self, laserdata: LaserData):
        self.laserdata = laserdata

    def __getitem__(self: LaserData, item):
        xmin, ymin, zmin, xmax, ymax, zmax = self.laserdata.bounds
        if len(item) == 2:
            x_slice, y_slice = item
            z_slice = slice(zmin, zmax)
        elif len(item) == 3:
            x_slice, y_slice, z_slice = item
        else:
            raise ValueError("Coordinate Based Index must be [xmin:xmax, ymin:ymax] or"
                             " [xmin:xmax, ymin:ymax, zmin:zmax]!")
        if type(x_slice) != slice or type(y_slice) != slice or type(z_slice) != slice:
            raise ValueError("Index must be slices!")
        xmin = x_slice.start if x_slice.start is not None else xmin
        ymin = y_slice.start if y_slice.start is not None else ymin
        zmin = z_slice.start if z_slice.start is not None else zmin
        xmax = x_slice.stop if x_slice.stop is not None else xmax
        ymax = y_slice.stop if y_slice.stop is not None else ymax
        zmax = z_slice.stop if z_slice.stop is not None else zmax

        x_in_range = np.logical_and(self.laserdata.x >= xmin, self.laserdata.x <= xmax)
        y_in_range = np.logical_and(self.laserdata.y >= ymin, self.laserdata.y <= ymax)
        z_in_range = np.logical_and(self.laserdata.z >= zmin, self.laserdata.z <= zmax)

        xy_in_range = np.logical_and(x_in_range, y_in_range)
        in_range = np.logical_and(xy_in_range, z_in_range)

        return self.laserdata[in_range]


def concatenate_laserdata(pointclouds):
    """
    Combine a number of LaserData objects into one. A new LaserData Object containing all points of the input LaserDatas
    will be returned.

    Author: Alexander Roß

    :param list of LaserData pointclouds: List of LaserData Objects to combine.
    :return: Combined LaserData object.
    """
    np_arrays = []
    for ld in pointclouds:
        np_arrays.append(ld.numpy())

    np_concat = np.concatenate(np_arrays)
    ld_concat = LaserData.from_numpy(np_concat)

    return ld_concat


# ----------------------------------------------------------------------------------------------------------------------
def file_id(filename, id_delimiter=None):
    """
    Returns the file-id extracted from a filename.
    Author: Alexander Roß

    :param str filename: Name of the file containing the id matching id_regex
    :param str or None id_delimiter: Symbols separating x- and y-coordinates in filename.
    :return: The id of the file
    :rtype:  str or None
    """
    id_delimiter = id_delimiter or "_"

    id_regex = f"([0-9]+{id_delimiter}[0-9]+)"  # Regex to extract tile-id from strings
    id_match = re.search(id_regex, filename)
    if id_match is not None:
        filename_id = id_match.group()
    else:
        filename_id = None

    return filename_id
# ----------------------------------------------------------------------------------------------------------------------
