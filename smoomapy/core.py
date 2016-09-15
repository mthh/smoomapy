#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
More or less a python port of Stewart method from R SpatialPositon package
(https://github.com/Groupe-ElementR/SpatialPosition/).

@author: mthh
"""
import numpy as np
import pyproj
from scipy.interpolate import griddata as scipy_griddata
from matplotlib.mlab import griddata as mlab_griddata
from matplotlib.pyplot import contourf
from shapely.geometry import Polygon, MultiPolygon, Point
from shapely.prepared import prep
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
        The span!
    beta : float
        The beta!
    typefct : str, optionnal
        The type of function in {"exponential", "pareto"} (default: "exponential").
    nb_class : int, default None
        The number of class, if unset will most likely be 8
        (default: None)
    resolution : int, optionnal
        The resolution to use (in unit of the input file), if not set a default
        resolution will be used in order to make a grid containing around
        7500 pts (default: None).
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
    distGeo = kwargs.get("distGeo", kwargs.get("longlat", True))
    StePot = SmoothStewart(input_geojson_points, variable_name, span,
                           beta, typefct, resolution,
                           None, mask, distGeo=distGeo)
    return StePot.render(nb_class=nb_class,
                         user_defined_breaks=user_defined_breaks,
                         output=output)


def make_regular_points_with_no_res(bounds, nb_points=7560):
    """
    Return a regular grid of points within `bounds` with the specified
    number of points (or a close approximate value).

    Parameters
    ----------
    bounds : 4-floats tuple
        The bbox of the grid, as xmin, ymin, xmax, ymax.
    nb_points : int, optionnal
        The desired number of points (default: 7560)

    Returns
    -------
    points : numpy.array
        An array of coordinates
    shape : 2-floats tuple
        The number of points on each dimension (width, height)
    """
    xmin, ymin, xmax, ymax = bounds
    nb_x = int(nb_points**0.5)
    nb_y = int(nb_points**0.5)

    height = (ymax-ymin) / (nb_y - 2)
    width = (xmax-xmin) / (nb_x - 2)

    x_start = xmin - width
    y_start = ymin - height

    prog_x = \
        [x_start + (width * i) for i in range(nb_x + 1)]
    prog_y = \
        [y_start + (height * i) for i in range(nb_y + 1)]

    return (np.array([(x, y) for x in prog_x for y in prog_y]),
            (len(prog_x), len(prog_y)))


def make_regular_points(bounds, resolution):
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
    xmin, ymin, xmax, ymax = bounds
    nb_x = int(round((xmax - xmin) / resolution))
    nb_y = int(round((ymax - ymin) / resolution))

    height = (ymax-ymin) / nb_y
    width = (xmax-xmin) / nb_x
    x_start = xmin - width
    y_start = ymin - height

    nb_x += 2
    nb_y += 2

    prog_x = \
        [x_start + (resolution * i) for i in range(nb_x + 1)]
    prog_y = \
        [y_start + (resolution * i) for i in range(nb_y + 1)]

    return (np.array([(x, y) for x in prog_x for y in prog_y]),
            (len(prog_x), len(prog_y)))


def make_dist_mat(xy1, xy2, longlat=False):
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
    if not longlat:
        d0 = np.subtract.outer(xy1[:, 0], xy2[:, 0])
        d1 = np.subtract.outer(xy1[:, 1], xy2[:, 1])
        return np.hypot(d0, d1)
    else:
        return hav_dist(xy1[:, None], xy2)


def hav_dist(locs1, locs2, k=np.pi/180):
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
    locs1 = locs1 * k
    locs2 = locs2 * k
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


class Idx:
    def __init__(self):
        self.values = []

    def add(self, value):
        self.values.append(value)
        return True


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
        7500 pts (default: None).
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
           func_grid="scipy", output="GeoJSON", new_mask=False)
        Render the contour polygon according to the choosen number of class and
        the choosen classification method (or according to
        `user_defined_breaks` which will overwrite these parameters)
    """

    def __init__(self, input_layer, variable_name, span, beta,
                 typefct='exponential', resolution=None,
                 variable_name2=None, mask=None, **kwargs):
        self.longlat = kwargs.get("distGeo", kwargs.get("longlat", True))
        self.gdf = input_layer if isinstance(input_layer, GeoDataFrame) \
            else GeoDataFrame.from_file(input_layer)
        self.proj_robinson = (
            """+proj=robin +lon_0=0 +x_0=0 +y_0=0 """
            """+ellps=WGS84 +datum=WGS84 +units=m +no_defs""")
        self.proj_nat_earth = """+proj=natearth"""
        self.info = (
            'SmoothStewart - variable : {}{} ({} features)\n'
            'beta : {} - span : {} - function : {}'
            ).format(variable_name,
                     " / {}".format(variable_name2) if variable_name2 else "",
                     len(self.gdf), beta, span, typefct)
        self.gdf.to_crs(crs="+proj=natearth", inplace=True)
        if mask is not None:
            if isinstance(mask, GeoDataFrame):
                self.mask = mask
            elif isinstance(mask, str) and isinstance(input_layer, str) \
                    and mask == input_layer:
                self.mask = self.gdf.copy()
            else:
                self.mask = GeoDataFrame.from_file(mask)

            self.mask.to_crs(crs="+proj=natearth", inplace=True)
            self.check_mask()
        else:
            self.use_mask = False

        # Approx. maximum extend we can reach with "nat-earth" projection :
        self.poly_max_extend = Polygon(
            [(-9602645.20918163, 9072201.505771412),
             (0.0, 9072201.505771412),
             (9602645.20918163, 9072201.505771412),
             (17446658.417140715, 0.0),
             (9602645.20918163, -9072201.505771412),
             (0.0, -9072201.518061588),
             (-9602645.20918163, -9072201.505771412),
             (-17446658.417140715, 0.0),
             (-9602645.20918163, 9072201.505771412)
             ])

        self.info2 = ""
        self.info3 = "Clipping mask: {}".format(self.use_mask)

        self.gdf.loc[:, variable_name] = self.gdf[variable_name].astype(float)
        self.gdf = self.gdf[self.gdf[variable_name].notnull()]

        if variable_name2:
            self.gdf.loc[:, variable_name2] = \
                self.gdf[variable_name2].astype(float)
            self.gdf = self.gdf[self.gdf[variable_name2].notnull()]

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

    def _compute_interact_density(self, matdist, typefun, beta, span):
        if 'pareto' in typefun:
            alpha = (2.0 ** (1.0 / beta) - 1.0) / span
            self.mat_dens = (1 + alpha * matdist) ** (-beta)
        elif 'exponential' in typefun:
            alpha = np.log(2) / span ** beta
            self.mat_dens = np.exp(- alpha * matdist ** beta)
        else:
            raise ValueError('Bad interaction function argument: {}'
                             .format(typefun))

    def compute_pot(self, variable_name, span, beta,
                    resolution=None, typefct="exponential",
                    variable_name2=None):
        knownpts = self.gdf

        if self.use_mask:
            bounds = self.mask.buffer(10).total_bounds
        else:
            tmp = np.max(
                [(knownpts.total_bounds[2] - knownpts.total_bounds[0])/10,
                 (knownpts.total_bounds[3] - knownpts.total_bounds[1])/10])
            tmp = span * 1.5 if tmp < span * 1.5 else tmp
            bounds = knownpts.buffer(tmp).total_bounds

        self.unknownpts, self.shape = make_regular_points(bounds, resolution) \
            if resolution else make_regular_points_with_no_res(bounds)

        if not self.longlat:
            knwpts_coords = np.array([
                (g.coords.xy[0][0], g.coords.xy[1][0])
                for g in knownpts.geometry.centroid])

            mat_dist = make_dist_mat(knwpts_coords,
                                     self.unknownpts,
                                     longlat=False)

        else:
            knwpts_coords = np.array([
                [g.coords.xy[1][0], g.coords.xy[0][0]]
                for g in knownpts.to_crs(
                    {"init": "epsg:4326"}).geometry.centroid])

            natearth = pyproj.Proj("+proj=natearth")
            wgs84 = pyproj.Proj("+init=EPSG:4326")

            prep_mask = prep(self.poly_max_extend)

            idx = Idx()
            self.geo_unknownpts = np.array([[i[1], i[0]] for n, i in enumerate(
                zip(*pyproj.transform(natearth, wgs84,
                                      [i[0] for i in self.unknownpts],
                                      [i[1] for i in self.unknownpts])))
                if prep_mask.contains(Point(i)) and idx.add(n)])

            self.unknownpts = self.unknownpts[idx.values]

            mat_dist = make_dist_mat(knwpts_coords,
                                     self.geo_unknownpts,
                                     longlat=True)

        self._compute_interact_density(mat_dist, typefct, beta, span)

        if variable_name2:
            self.pot1 = (
                knownpts[variable_name].values[:, np.newaxis] * self.mat_dens
                ).sum(axis=0)
            self.pot2 = (
                knownpts[variable_name2].values[:, np.newaxis] * self.mat_dens
                ).sum(axis=0)
            self.pot = (np.true_divide(self.pot1, self.pot2)).round(8)
            _nan_mask = np.argwhere(~np.isnan(self.pot)).reshape(-1)
            self.pot = self.pot[_nan_mask]
            self.unknownpts = self.unknownpts[_nan_mask]

        else:
            self.pot = (
                knownpts[variable_name].values[:, np.newaxis] * self.mat_dens
                ).sum(axis=0).round(8)

        self.x = np.array([c[0] for c in self.unknownpts])
        self.y = np.array([c[1] for c in self.unknownpts])

        self.xi = np.linspace(np.nanmin(self.x), np.nanmax(self.x),
                              self.shape[0] if self.shape[0] > 100 else 100)
        self.yi = np.linspace(np.nanmin(self.y), np.nanmax(self.y),
                              self.shape[1] if self.shape[0] > 100 else 100)

        self.info2 = ("unknown points : {} - interpolation grid shape : {}"
                      ).format(len(self.unknownpts), self.shape)

    def change_interp_grid_shape(self, new_shape):
        self.xi = np.linspace(np.nanmin(self.x),
                              np.nanmax(self.x),
                              new_shape[0])
        self.yi = np.linspace(np.nanmin(self.y),
                              np.nanmax(self.y),
                              new_shape[1])

    def define_levels(self, nb_class, disc_func):
        pot = self.pot
        _min = np.nanmin(pot)
        if not nb_class:
            nb_class = int(get_opt_nb_class(len(self.pot)) - 2)

        if not disc_func or "prog_geom" in disc_func:
            levels = [_min] + [
                pot.max() / i for i in range(1, nb_class + 1)][::-1]
        elif "equal_interval" in disc_func:
            _bin = pot.max() / nb_class
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
                .intersection({"Polygon", "MultiPolygon"})) > 0 \
                and self.gdf.crs == self.mask.crs:
            self.use_mask = True
        else:
            self.use_mask = False

    def render(self, nb_class=8, disc_func=None, user_defined_breaks=None,
               func_grid="scipy", output="GeoJSON", new_mask=False):
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
        func_grid : str, optionnal
            The kind of function to use to make the interpolation
            ("scipy" for scipy.interpolate.griddata,
             "matplotlib" for matplotlib.mlab.griddata,
             "rbf" for scipy.interpolate.rbf, default: "scipy").
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
            self.mask = new_mask.to_crs(crs="+proj=natearth")
            self.check_mask()
        elif isinstance(new_mask, (str, BytesIO, StringIO)):
            self.use_mask = True
            self.mask = GeoDataFrame.from_file(
                new_mask).to_crs(crs="+proj=natearth")
            self.check_mask()

        if func_grid == "scipy":
            self.zi = scipy_griddata((self.x, self.y), pot,
                                     (self.xi[None, :], self.yi[:, None]),
                                     method='linear').round(8)
        elif func_grid == "scipy-cubic":
            self.zi = scipy_griddata((self.x, self.y), pot,
                                     (self.xi[None, :], self.yi[:, None]),
                                     method='cubic').round(8)
        elif func_grid == "scipy-nearest":
            self.zi = scipy_griddata((self.x, self.y), pot,
                                     (self.xi[None, :], self.yi[:, None]),
                                     method='nearest').round(8)
        elif func_grid == "matplotlib":
            self.zi = mlab_griddata(self.x, self.y, pot,
                                    self.xi, self.yi, interp='linear'
                                    ).round(8)
        else:
            raise ValueError("Invalid interpolation function name provided")

        if user_defined_breaks:
            levels = user_defined_breaks
        else:
            levels = self.define_levels(nb_class, disc_func)

        collec_poly = contourf(
            self.xi, self.yi, self.zi,
            levels,
            vmax=abs(np.nanmax(self.zi)), vmin=-abs(np.nanmin(self.zi)))
        levels = collec_poly.levels
        levels[-1] = np.nanmax(pot)
        res = isopoly_to_gdf(collec_poly, levels=levels[1:], field_name="max")

        res.crs = self.gdf.crs
        res["min"] = [np.nanmin(self.pot)] + res["max"][0:len(res)-1].tolist()
        res["center"] = (res["min"] + res["max"]) / 2

        ix_max_ft = len(res) - 1
        if self.use_mask:
            res.loc[0:ix_max_ft, "geometry"] = res.geometry.buffer(
                0).intersection(unary_union(self.mask.geometry.buffer(0)))

        res.loc[0:ix_max_ft, "geometry"] = res.geometry.buffer(
            0).intersection(self.poly_max_extend.buffer(-0.1))
        # Repair geometries if necessary :
        if not all(t in ("MultiPolygon", "Polygon") for t in res.geom_type):
            res.loc[0:ix_max_ft, "geometry"] = \
                [geom if geom.type in ("Polygon", "MultiPolygon")
                 else MultiPolygon(
                     [j for j in geom if j.type in ('Polygon', 'MultiPolygon')]
                     )
                 for geom in res.geometry]

        res.to_crs({'init': 'epsg:4326'}, inplace=True)

        return res.to_json().encode() if "geojson" in output.lower() else res
