#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
More or less a python port of Stewart method from R SpatialPositon package
(https://github.com/Groupe-ElementR/SpatialPosition/).

@author: mthh
"""
import numpy as np
from matplotlib.pyplot import contourf
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
from geopandas import GeoDataFrame
from io import BytesIO, StringIO
try:
    from jenkspy import jenks_breaks
except:
    jenks_breaks = None
from .helpers_classif import get_opt_nb_class, maximal_breaks, head_tail_breaks


def quick_stewart(input_geojson_points, variable_name, span,
                  beta=2, typefct='exponential',
                  nb_class=None, resolution=None, mask=None,
                  user_defined_breaks=None, variable_name2=None,
                  output="GeoJSON", **kwargs):
    """
    Main function, acting as a one-shot wrapper around SmoothStewart object.
    Read a file of point values and optionnaly a mask file,
    return the smoothed representation as GeoJSON or GeoDataFrame.

    Parameters
    ----------
    input_geojson_points : str
        Path to file to use as input (Points/Polygons) or GeoDataFrame object,
        must contains a relevant numerical field.
    variable_name : str
        The name of the variable to use (numerical field only).
    span : int
        The span (meters).
    beta : float
        The beta!
    typefct : str, optionnal
        The type of function in {"exponential", "pareto"} (default: "exponential").
    nb_class : int, default None
        The number of class, if unset will most likely be 8
        (default: None)
    resolution : int, optionnal
        The resolution to use (in meters), if not set a default
        resolution will be used in order to make a grid containing around
        10000 pts (default: None).
    mask : str, optionnal
        Path to the file (Polygons only) to use as clipping mask,
        can also be a GeoDataFrame (default: None).
    user_defined_breaks : list or tuple, optionnal
        A list of ordered break to use to construct the contours
        (override `nb_class` value if any, default: None).
    variable_name2 : str, optionnal
        The name of the 2nd variable to use (numerical field only); values
        computed from this variable will be will be used as to divide
        values computed from the first variable (default: None)
    output : string, optionnal
        The type of output expected (not case-sensitive)
        in {"GeoJSON", "GeoDataFrame"} (default: "GeoJSON").

    Returns
    -------
    smoothed_result : bytes or GeoDataFrame,
        The result, dumped as GeoJSON (utf-8 encoded) or as a GeoDataFrame.


    Examples
    --------
    Basic usage, output to raw geojson (bytes):

    >>> result = quick_stewart("some_file.geojson", "some_variable",
                               span=12500, beta=3, typefct="exponential")

    More options, returning a GeoDataFrame:

    >>> smooth_gdf = quick_stewart("some_file.geojson", "some_variable",
                                   span=12500, beta=3, typefct="pareto",
                                   output="GeoDataFrame")
    """
    StePot = SmoothStewart(input_geojson_points, variable_name, span,
                           beta, typefct, resolution, None, mask, **kwargs)
#                           None, mask, distGeo=distGeo)
    return StePot.render(nb_class=nb_class,
                         user_defined_breaks=user_defined_breaks,
                         output=output)


def make_regular_points_with_no_res(bounds, nb_points=10000):
    """
    Return a regular grid of points within `bounds` with the specified
    number of points (or a close approximate value).

    Parameters
    ----------
    bounds : 4-floats tuple
        The bbox of the grid, as xmin, ymin, xmax, ymax.
    nb_points : int, optionnal
        The desired number of points (default: 10000)

    Returns
    -------
    points : numpy.array
        An array of coordinates
    shape : 2-floats tuple
        The number of points on each dimension (width, height)
    """
    minlon, minlat, maxlon, maxlat = bounds
    minlon, minlat, maxlon, maxlat = bounds
    offset_lon = (maxlon - minlon) / 8
    offset_lat = (maxlat - minlat) / 8
    minlon -= offset_lon
    maxlon += offset_lon
    minlat -= offset_lat
    maxlat += offset_lat

    nb_x = int(nb_points**0.5)
    nb_y = int(nb_points**0.5)

    return (
        np.linspace(minlon, maxlon, nb_x),
        np.linspace(minlat, maxlat, nb_y),
        (nb_y, nb_x)
        )


def make_regular_points(bounds, resolution, longlat=True):
    """
    Return a regular grid of points within `bounds` with the specified
    resolution.

    Parameters
    ----------
    bounds : 4-floats tuple
        The bbox of the grid, as xmin, ymin, xmax, ymax.
    resolution : int
        The resolution to use, in the same unit as `bounds`

    Returns
    -------
    points : numpy.array
        An array of coordinates
    shape : 2-floats tuple
        The number of points on each dimension (width, height)
    """
#    xmin, ymin, xmax, ymax = bounds
    minlon, minlat, maxlon, maxlat = bounds
    offset_lon = (maxlon - minlon) / 8
    offset_lat = (maxlat - minlat) / 8
    minlon -= offset_lon
    maxlon += offset_lon
    minlat -= offset_lat
    maxlat += offset_lat

    if longlat:
        height = hav_dist(
                np.array([(maxlon + minlon) / 2, minlat]),
                np.array([(maxlon + minlon) / 2, maxlat])
                )
        width = hav_dist(
                np.array([minlon, (maxlat + minlat) / 2]),
                np.array([maxlon, (maxlat + minlat) / 2])
                )
    else:
        height = np.linalg.norm(
            np.array([(maxlon + minlon) / 2, minlat])
            - np.array([(maxlon + minlon) / 2, maxlat]))
        width = np.linalg.norm(
            np.array([minlon, (maxlat + minlat) / 2])
            - np.array([maxlon, (maxlat + minlat) / 2]))

    nb_x = int(round(width / resolution))
    nb_y = int(round(height / resolution))
    if nb_y * 0.6 > nb_x:
        nb_x = int(nb_x + nb_x / 3)
    elif nb_x * 0.6 > nb_y:
        nb_y = int(nb_y + nb_y / 3)
    return (
        np.linspace(minlon, maxlon, nb_x),
        np.linspace(minlat, maxlat, nb_y),
        (nb_y, nb_x)
        )


def _compute_centroids(geometries):
    res = []
    for geom in geometries:
        if hasattr(geom, '__len__'):
            ix_biggest = np.argmax([g.area for g in geom])
            res.append(geom[ix_biggest].centroid)
        else:
            res.append(geom.centroid)
    return res


def make_dist_mat(xy1, xy2, longlat=True):
    """
    Return a distance matrix between two set of coordinates.
    Use geometric distance (default) or haversine distance (if longlat=True).

    Parameters
    ----------
    xy1 : numpy.array
        The first set of coordinates as [(x, y), (x, y), (x, y)].
    xy2 : numpy.array
        The second set of coordinates as [(x, y), (x, y), (x, y)].
    longlat : boolean, optionnal
        Whether the coordinates are in geographic (longitude/latitude) format
        or not (default: False)

    Returns
    -------
    mat_dist : numpy.array
        The distance matrix between xy1 and xy2
    """
    if longlat:
        return hav_dist(xy1[:, None], xy2)
    else:
        d0 = np.subtract.outer(xy1[:, 0], xy2[:, 0])
        d1 = np.subtract.outer(xy1[:, 1], xy2[:, 1])
        return np.hypot(d0, d1)


def hav_dist(locs1, locs2):
    """
    Return a distance matrix between two set of coordinates.
    Use geometric distance (default) or haversine distance (if longlat=True).

    Parameters
    ----------
    locs1 : numpy.array
        The first set of coordinates as [(long, lat), (long, lat)].
    locs2 : numpy.array
        The second set of coordinates as [(long, lat), (long, lat)].

    Returns
    -------
    mat_dist : numpy.array
        The distance matrix between locs1 and locs2
    """
    locs1 = np.radians(locs1)
    locs2 = np.radians(locs2)
    cos_lat1 = np.cos(locs1[..., 0])
    cos_lat2 = np.cos(locs2[..., 0])
    cos_lat_d = np.cos(locs1[..., 0] - locs2[..., 0])
    cos_lon_d = np.cos(locs1[..., 1] - locs2[..., 1])
    return 6367000 * np.arccos(
        cos_lat_d - cos_lat1 * cos_lat2 * (1 - cos_lon_d))


def isopoly_to_gdf(collec_poly, levels, field_name="levels"):
    """
    Convert a collection of matplotlib.contour.QuadContourSet to a GeoDataFrame
    Set an attribute `field_name` on each feature, according to `levels` values
    (`levels` must have the same number of features as the collection of contours)

    Parameters
    ----------
    collection_polygons : matplotlib.contour.QuadContourSet
        The result of a grid interpolation from matplotlib.
    levels : array-like
        The value to use as attributes for the constructed GeoDataFrame.
    field_name : str
        The name of the field to be fill by values contained in
        `levels` variable (default: "levels").

    Returns
    -------
    gdf_contours : GeoDataFrame
        The result as a GeoDataFrame.
    """
    polygons, data = [], []

    for i, polygon in enumerate(collec_poly.collections):
        mpoly = []
        for path in polygon.get_paths():
            path.should_simplify = False
            poly = path.to_polygons()
            exterior, holes = [], []
            if len(poly) > 0 and len(poly[0]) > 3:
                exterior = poly[0]
                if len(poly) > 1:
                    holes = [h for h in poly[1:] if len(h) > 3]
            mpoly.append(Polygon(exterior, holes))
        if len(mpoly) > 1:
            mpoly = MultiPolygon(mpoly)
            polygons.append(mpoly)
            data.append(levels[i])
        elif len(mpoly) == 1:
            polygons.append(mpoly[0])
            data.append(levels[i])

    return GeoDataFrame(geometry=polygons,
                        data=data,
                        columns=[field_name])


class SmoothStewart:
    """
    Main object, allowing to create an instance with some required parameters
    (span, beta, etc.) then render the contour polygons according to various
    parameters (data classification, number of bins, output format, etc.)

    Parameters
    ----------
    input_layer : str
        Path to file to use as input (Points/Polygons) or GeoDataFrame object,
        must contains a relevant numerical field.
    variable_name : str
        The name of the variable to use (numerical field only).
    span : int
        The span!
    beta : float
        The beta!
    typefct : str, optionnal
        The type of function in {"exponential", "pareto"} (default: "exponential").
    resolution : int, optionnal
        The resolution to use (in unit of the input file), if not set a default
        resolution will be used in order to make a grid containing around
        10000 pts (default: None).
    mask : str, optionnal
        Path to the file (Polygons only) to use as clipping mask (default: None).
    variable_name2 : str, optionnal
        The name of the 2nd variable to use (numerical field only); values
        computed from this variable will be will be used as to divide
        values computed from the first variable (default: None)

    Attributes
    ----------
    pot : numpy.ndarray
        The computed potential values for each `unknownpts`.
    unknownpts : numpy.ndarray
        The coordinates for each unknown points, on which values
        have been computed according to the choosen model (span, beta, etc.)

    Methods
    -------
    render(nb_class=8, disc_func=None, user_defined_breaks=None,
           output="GeoJSON", new_mask=False)
        Render the contour polygon according to the choosen number of class and
        the choosen classification method (or according to
        `user_defined_breaks` which will overwrite these parameters)
    """

    def __init__(self, input_layer, variable_name, span, beta,
                 typefct='exponential', resolution=None,
                 variable_name2=None, mask=None, **kwargs):
        self.longlat = kwargs.get("distGeo", kwargs.get("longlat", True))
        self.proj_to_use = {'init': 'epsg:4326'} if self.longlat \
            else kwargs.get("projDistance", None) \
            or ("""+proj=robin +lon_0=0 +x_0=0 +y_0=0 """
                """+ellps=WGS84 +datum=WGS84 +units=m +no_defs""")

        self.gdf = input_layer.copy() if isinstance(input_layer, GeoDataFrame) \
            else GeoDataFrame.from_file(input_layer)

        if self.gdf.crs and self.gdf.crs is not self.proj_to_use:
            self.gdf.to_crs(self.proj_to_use, inplace=True)
        else:
            self.gdf.crs = self.proj_to_use

        self.info = (
            'SmoothStewart - variable : {}{} ({} features)\n'
            'beta : {} - span : {} - function : {}'
            ).format(variable_name,
                     " / {}".format(variable_name2) if variable_name2 else "",
                     len(self.gdf), beta, span, typefct)
        if mask is not None:
            if isinstance(mask, GeoDataFrame):
                self.mask = mask
            elif isinstance(mask, str) and isinstance(input_layer, str) \
                    and mask == input_layer:
                self.mask = self.gdf.copy()
            else:
                self.mask = GeoDataFrame.from_file(mask)

            self.check_mask()
        else:
            self.use_mask = False

        self.info2 = ""
        self.info3 = "Clipping mask: {}".format(self.use_mask)

        # Convert the first value field to a numeric field if not already,
        # and dont take into account features with no value / NaN value
        if not self.gdf[variable_name].dtype in (float, int):
            self.gdf.loc[:, variable_name] = \
                self.gdf[variable_name].replace('', np.NaN)

        self.gdf.loc[:, variable_name] = self.gdf[variable_name].astype(float)
        self.gdf = self.gdf[self.gdf[variable_name].notnull()]

        # Convert the second value field to a numeric field if not already,
        # and dont take into account features with no value / NaN value
        if variable_name2:
            if not self.gdf[variable_name2].dtype in (float, int):
                self.gdf.loc[:, variable_name2] = \
                    self.gdf[variable_name2].replace('', np.NaN)
            self.gdf.loc[:, variable_name2] = \
                self.gdf[variable_name2].astype(float)
            self.gdf = self.gdf[self.gdf[variable_name2].notnull()]

        # Provide a new index if entries have been removed :
        self.gdf.index = range(len(self.gdf))

        # Calculate the value for each unknown points of the grid :
        self.compute_pot(variable_name, span, beta,
                         variable_name2=variable_name2,
                         resolution=resolution,
                         typefct=typefct)

    def __repr__(self):
        return "\n".join([self.info, self.info2, self.info3])

    def __str__(self):
        return "\n".join([self.info, self.info2, self.info3])

    @property
    def properties(self):
        print("\n".join([self.info, self.info2, self.info3]))

    @staticmethod
    def _compute_interact_density(matdist, typefun, beta, span):
        if 'pareto' in typefun:
            alpha = (2.0 ** (1.0 / beta) - 1.0) / span
            return (1 + alpha * matdist) ** (-beta)
        elif 'exponential' in typefun:
            alpha = np.log(2) / span ** beta
            return np.exp(- alpha * matdist ** beta)
        else:
            raise ValueError('Bad interaction function argument: {}'
                             .format(typefun))

    def compute_pot(self, variable_name, span, beta,
                    resolution=None, typefct="exponential",
                    variable_name2=None):
        knownpts = self.gdf

        if self.use_mask:
            bounds = self.mask.total_bounds
        else:
            bounds = knownpts.total_bounds

        # Get the x and y axis of the grid:
        self.XI, self.YI, self.shape = make_regular_points(bounds, resolution) \
            if resolution else make_regular_points_with_no_res(bounds)

        # Compute the coordinates of each point of the grid :
        unknownpts = np.array([(x, y) for x in self.XI for y in self.YI])

        # Use the centroid if the feature is a Polygon
        #  or use the centroid of the largest Polygon for a MultiPolygon:
        if all(i in ("Polygon", "Point") for i in knownpts.geom_type.values):
            centroids = knownpts.geometry.centroid 
        else:
            centroids = _compute_centroids(knownpts.geometry)

        # Coordinates of every known point:
        knwpts_coords = np.array([
            (g.coords.xy[0][0], g.coords.xy[1][0])
            for g in centroids])

        # Compute the interaction matrix: 
        mat_dens = self._compute_interact_density(
                make_dist_mat(knwpts_coords, unknownpts, longlat=self.longlat),
                typefct, beta, span)

        if not variable_name2:
            self.pot = (
                knownpts[variable_name].values[:, np.newaxis] * mat_dens
                ).sum(axis=0).round(8)
        else:
            self.pot1 = (
                knownpts[variable_name].values[:, np.newaxis] * mat_dens
                ).sum(axis=0)
            self.pot2 = (
                knownpts[variable_name2].values[:, np.newaxis] * mat_dens
                ).sum(axis=0)
            self.pot = (np.true_divide(self.pot1, self.pot2)).round(8)

            # Replace NaN values by -1.0 :
            self.pot[np.argwhere(np.isnan(self.pot)).reshape(-1)] = -1.0
            # Replace inf values by -1.0 :
            self.pot[np.argwhere(np.isinf(self.pot)).reshape(-1)] = -1.0

        self.info2 = ("unknown points : {} - interpolation grid shape : {}"
                      ).format(len(unknownpts), self.shape)

    def define_levels(self, nb_class, disc_func):
        pot = self.pot
        _min = np.nanmin(pot)

        if not nb_class:
            nb_class = int(get_opt_nb_class(len(pot)) - 2)
        if not disc_func or "prog_geom" in disc_func:
            levels = [_min] + [
                np.nanmax(pot) / i for i in range(1, nb_class + 1)][::-1]
        elif "equal_interval" in disc_func:
            _bin = np.nanmax(pot) / nb_class
            levels = [_min] + [_bin * i for i in range(1, nb_class+1)]
        elif "percentiles" in disc_func:
            levels = np.percentile(
                np.concatenate((pot[pot.nonzero()], np.array([_min]))),
                np.linspace(0.0, 100.0, nb_class+1))
        elif "jenks" in disc_func:
            levels = list(jenks_breaks(np.concatenate(
                ([_min], pot[pot.nonzero()])), nb_class))
            levels[0] = levels[0] - _min * 0.01
        elif "head_tail" in disc_func:
            levels = head_tail_breaks(np.concatenate(
                ([_min], pot[pot.nonzero()])))
        elif "maximal_breaks" in disc_func:
            levels = maximal_breaks(np.concatenate(
                ([_min], pot[pot.nonzero()])), nb_class)
        else:
            raise ValueError

        return levels

    def check_mask(self):
        if len(set(self.mask.type)
                .intersection({"Polygon", "MultiPolygon"})) > 0:
            if self.mask.crs and self.mask.crs is not self.proj_to_use:
                self.mask.to_crs(self.proj_to_use, inplace=True)
            else:
                self.mask.crs = self.proj_to_use

            self.use_mask = True
        else:
            self.use_mask = False

    def render(self, nb_class=8, disc_func=None, user_defined_breaks=None,
               output="GeoJSON", new_mask=False):
        """
        Parameters
        ----------
        nb_class : int, optionnal
            The number of class (default: 8).
        disc_func : str, optionnal
            The kind of data classification to be used (to be choosed in
            "equal_interval", "jenks", "percentiles, "head_tail_breaks"
            and "prog_geom"), default: None.
        user_defined_breaks : list or tuple, optionnal
            A list of ordered break to use to construct the contours
            (override `nb_class` and `disc_func` values if any)
            (default: None).
        output : string, optionnal
            The type of output expected (not case-sensitive)
            in {"GeoJSON", "GeoDataFrame"} (default: "GeoJSON").
        new_mask : str, optionnal
            Use a new mask by giving the path to the file (Polygons only)
            to use as clipping mask, can also be directly a GeoDataFrame
            (default: False).

        Returns
        -------
        smoothed_result : bytes or GeoDataFrame
            The result, dumped as GeoJSON (utf-8 encoded) or as a GeoDataFrame.
        """
        if disc_func and 'jenks' in disc_func and not jenks_breaks:
            raise ValueError(
                "Missing jenkspy package - could not use jenks breaks")

        pot = self.pot

        if new_mask is None:
            self.use_mask = False
            self.mask = None
        elif isinstance(new_mask, GeoDataFrame):
            self.use_mask = True
            self.mask = new_mask
            self.check_mask()
        elif isinstance(new_mask, (str, BytesIO, StringIO)):
            self.use_mask = True
            self.mask = GeoDataFrame.from_file(new_mask)
            self.check_mask()

        if user_defined_breaks:
            levels = user_defined_breaks
            if levels[len(levels) - 1] < np.nanmax(pot):
                levels = levels + [np.nanmax(pot)]
            if levels[0] > np.nanmin(pot):
                levels = [np.nanmin(pot)] + levels
        else:
            levels = self.define_levels(nb_class, disc_func)

        # Ensure that the levels are unique/increasing
        #  to avoid error from `contourf` :
        s_levels = set(levels)
        if len(s_levels) != len(levels):
            levels = list(s_levels)
        levels.sort()

        try:
            collec_poly = contourf(
                self.XI, self.YI,
                self.pot.reshape(tuple(reversed(self.shape))).T,
                levels,
                vmax=abs(np.nanmax(self.pot)), vmin=-abs(np.nanmin(self.pot)))
        # Retry without setting the levels :
        except ValueError:
            collec_poly = contourf(
                self.XI, self.YI,
                self.pot.reshape(tuple(reversed(self.shape))).T,
                vmax=abs(np.nanmax(self.pot)), vmin=-abs(np.nanmin(self.pot)))

        levels = collec_poly.levels
        levels[-1] = np.nanmax(pot)
        res = isopoly_to_gdf(collec_poly, levels=levels[1:], field_name="max")

        res.crs = self.proj_to_use
        res["min"] = [np.nanmin(self.pot)] + res["max"][0:len(res)-1].tolist()
        res["center"] = (res["min"] + res["max"]) / 2

        ix_max_ft = len(res) - 1
        if self.use_mask:
            res.loc[0:ix_max_ft, "geometry"] = res.geometry.buffer(
                0).intersection(unary_union(self.mask.geometry.buffer(0)))

        # res.loc[0:ix_max_ft, "geometry"] = res.geometry.buffer(
        #     0).intersection(self.poly_max_extend.buffer(-0.1))
        # Repair geometries if necessary :
        if not all(t in ("MultiPolygon", "Polygon") for t in res.geom_type):
            res.loc[0:ix_max_ft, "geometry"] = \
                [geom if geom.type in ("Polygon", "MultiPolygon")
                 else MultiPolygon(
                     [j for j in geom if j.type in ('Polygon', 'MultiPolygon')]
                     )
                 for geom in res.geometry]

        if "geojson" in output.lower():
            return res.to_crs({"init": "epsg:4326"}).to_json().encode()
        else:
            return res
