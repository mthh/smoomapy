# -*- coding: utf-8 -*-
"""
Most likely a python port of Stewart method
from R SpatialPositon package (https://github.com/Groupe-ElementR/SpatialPosition/)
Allow to set a desired number of class or directly some custom breaks values.

@author: mthh
"""
import numpy as np
from matplotlib.mlab import griddata
from matplotlib.pyplot import contourf
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
from geopandas import GeoDataFrame


def quick_stewart(input_geojson_points, variable_name, span,
                 beta=2, typefct='exponential',
                 nb_class=None, resolution=None, mask=None,
                 user_defined_breaks=None, output="GeoJSON"):
    """
    Main function, read a file of point values and optionnaly a mask file,
    return the smoothed representation as GeoJSON.

    Parameters
    ----------
    input_geojson_points: str
        Path to file to use as input (Points/Polygons), must contains
        a relevant numerical field.
    variable_name: str
        The name of the variable to use (numerical field only).
    span: int
        The span!
    beta: float
        The beta!
    typefct: str, default "exponential"
        The type of function in {"exponential", "pareto"}
    nb_class: int, default None
        The number of class, if unset will most likely be 8.
    resolution: int, default None
        The resolution to use (in unit of the input file), if not set a resolution
        will be used in order to make a grid containing around 7560 pts.
    mask: str, default None
        Path to the file (Polygons only) to use as clipping mask.
    user_defined_breaks: list or tuple, default None
        A list of ordered break to use to construct the contours
        (override `nb_class` value if any)
    output: string, optionnal
        The type of output expected (not case-sensitive) in {"GeoJSON", "GeoDataFrame"}
        (default: "GeoJSON")

    Returns
    -------
    smoothed_geojson: bytes,
        The result dumped as GeoJSON (utf-8 encoded)


    Examples
    --------
    Basic usage, output to raw geojson (bytes):

    >>> result = quick_stewart("some_file.geojson", "some_variable",
                               span=12500, beta=3, typefct="exponential")

    Mote options, returning a GeoDataFrame:

    >>> smooth_gdf = quick_stewart("some_file.geojson", "some_variable",
                                   span=12500, beta=3, typefct="pareto",
                                   output="GeoDataFrame")
    """
    gdf = GeoDataFrame.from_file(input_geojson_points)

    if mask:
        mask = GeoDataFrame.from_file(mask) \
                if mask != input_geojson_points else gdf

        if len(set(gdf.type).intersection({"Polygon", "MultiPolygon"})) > 0 \
                and gdf.crs == mask.crs:
            use_mask = True
        else:
            print("Warning: Mask layer have to be (Multi)Polygon geometries"
                  " and use the same CRS as input values")
            use_mask = False

    pot, unknownpts, shape = compute(gdf,
                                     variable_name,
                                     span=span,
                                     beta=beta,
                                     resolution=resolution,
                                     typefct='exponential',
                                     mask=mask if use_mask else None)

    result = render_stewart(
        pot, unknownpts, nb_class if nb_class else 8, mask, shape,
        user_defined_breaks)
    result.crs = gdf.crs
    return result.to_crs({'init': 'epsg:4326'}).to_json().encode() \
        if "geojson" in output.lower() \
        else result.to_crs({'init': 'epsg:4326'})


def make_regular_points_with_no_res(bounds, nb_points=7560):
    """
    Return a regular grid of points within `bounds` with the specified
    number of points (or a close approximate value).

    Parameters
    ----------
    bounds: 4-floats tuple
        The bbox of the grid, as xmin, ymin, xmax, ymax.
    nb_points: int, optionnal
        The desired number of points (default: 7560)

    Returns
    -------
    points: numpy.array
        An array of coordinates
    shape: 2-floats tuple
        The number of points on each dimension (width, height)
    """
    xmin, ymin, xmax, ymax = bounds
    rows = int(nb_points**0.5)
    cols = int(nb_points**0.5)
    height = (ymax-ymin) / rows
    width = (xmax-xmin) / cols
    x_left_origin = xmin
    x_right_origin = xmin + width
    y_top_origin = ymax
    y_bottom_origin = ymax - height

    res_geoms = []
    for countcols in range(cols):
        y_top = y_top_origin
        y_bottom = y_bottom_origin
        for countrows in range(rows):
            res_geoms.append([
                (x_left_origin + x_right_origin) / 2, (y_top + y_bottom) / 2
                ])
            y_top = y_top - height
            y_bottom = y_bottom - height
        x_left_origin = x_left_origin + width
        x_right_origin = x_right_origin + width

    return (np.array(res_geoms), (cols, rows))


def make_regular_points(bounds, resolution):
    """
    Return a regular grid of points within `bounds` with the specified
    resolution.

    Parameters
    ----------
    bounds: 4-floats tuple
        The bbox of the grid, as xmin, ymin, xmax, ymax.
    resolution: int
        The resolution to use, in the same unit as `bounds`

    Returns
    -------
    points: numpy.array
        An array of coordinates
    shape: 2-floats tuple
        The number of points on each dimension (width, height)
    """
    xmin, ymin, xmax, ymax = bounds
    nb_x = int(round((xmax - xmin) / resolution + ((xmax - xmin) / resolution) / 10))
    nb_y = int(round((ymax - ymin) / resolution + ((ymax - ymin) / resolution) / 10))

    try:
        prog_x = \
            [(xmin - (xmax - xmin) / 20) + resolution * i for i in range(nb_x + 1)]
        prog_y = \
            [(ymin - (ymax - ymin) / 20) + resolution * i for i in range(nb_y + 1)]
    except ZeroDivisionError:
        raise ZeroDivisionError(
            'Please choose a finest resolution (by lowering the value of the '
            'resolution argument and/or providing an appropriate mask layer')
    return (np.array([(x, y) for x in prog_x for y in prog_y]),
            (len(prog_x), len(prog_y)))


def make_dist_mat(xy1, xy2, longlat=False):
    """
    Return a distance matrix between two set of coordinates.
    Use geometric distance (default) or haversine distance (if longlat=True).

    Parameters
    ----------
    xy1: numpy.array
        The first set of coordinates as [(x, y), (x, y), (x, y)].
    xy2: numpy.array
        The second set of coordinates as [(x, y), (x, y), (x, y)].
    longlat: boolean, optionnal
        Whether the coordinates are in geographic (longitude/latitude) format
        or not (default: False)

    Returns
    -------
    mat_dist: numpy.array
        The distance matrix between xy1 and xy2
    """
    if not longlat:
        d0 = np.subtract.outer(xy1[:, 0], xy2[:, 0])
        d1 = np.subtract.outer(xy1[:, 1], xy2[:, 1])
        return np.hypot(d0, d1)
    else:
        return hav_dist(xy1, xy2)


def hav_dist(locs1, locs2, k=np.pi/180):
    """
    Return a distance matrix between two set of coordinates.
    Use geometric distance (default) or haversine distance (if longlat=True).

    Parameters
    ----------
    locs1: numpy.array
        The first set of coordinates as [(long, lat), (long, lat)].
    locs2: numpy.array
        The second set of coordinates as [(long, lat), (long, lat)].

    Returns
    -------
    mat_dist: numpy.array
        The distance matrix between locs1 and locs2
    """
    locs1 = locs1 * k
    locs2 = locs2 * k
    cos_lat1 = np.cos(locs1[..., 1])
    cos_lat2 = np.cos(locs2[..., 1])
    cos_lat_d = np.cos(locs1[..., 1] - locs2[..., 1])
    cos_lon_d = np.cos(locs1[..., 0] - locs2[..., 0])
    return 6367 * np.arccos(cos_lat_d - cos_lat1 * cos_lat2 * (1 - cos_lon_d))


def compute_interact_density(matdist, typefun, beta, span):
    if 'pareto' in typefun:
        alpha = (2 ** (1 / beta) - 1) / span
        matDens = (1 + alpha * matdist) ** (-beta)
    elif 'exponential' in typefun:
        alpha = np.log(2) / span ** beta
        matDens = np.exp(- alpha * matdist ** beta)
    else:
        raise ValueError('Bad interaction function argument: {}'
                         .format(typefun))
    return matDens.round(8)


#def compute_potentials(matopport):
#    return matopport.sum(axis=0)
#
#def compute_opportunity(knownpts, matdens, varname):
#    matOpport = knownpts[varname].values[:, np.newaxis] * matdens
#    return matOpport.round(8)
#
#def compute_opportunity_potentials(knownpts_values, matdens):
#    matOpport = knownpts_values[:, np.newaxis] * matdens
#    return matOpport.sum(axis=0).round(8)

def render_stewart(pot, unknownpts, nb_class=8, mask=None, shape=None,
                   user_defined_breaks=None):
    x = np.array([c[0] for c in unknownpts])
    y = np.array([c[1] for c in unknownpts])
    xi = np.linspace(np.nanmin(x), np.nanmax(x), shape[0])
    yi = np.linspace(np.nanmin(y), np.nanmax(y), shape[1])
    zi = griddata(x, y, pot, xi, yi, interp='linear').round(8)

    levels = [0] + [pot.max()/i for i in range(1, nb_class + 1)][::-1] \
        if not user_defined_breaks else user_defined_breaks
    collec_poly = contourf(
        xi, yi, zi,
        levels,
        vmax=abs(zi).max(), vmin=-abs(zi).max())

    levels = collec_poly.levels
    levels[-1] = np.nanmax(pot)
    res = isopoly_to_gdf(collec_poly, levels=levels[1:], field_name="max")
    res["min"] = [0] + [res["max"][i-1] for i in range(1, len(res))]
    res["center"] = res["min"] + res["max"] / 2
    if mask is not None:
        res.geometry = res.geometry.buffer(0).intersection(
                                        unary_union(mask.geometry.buffer(0)))
    return res


def isopoly_to_gdf(collec_poly, levels, field_name="levels"):
    """
    Convert a collection of matplotlib.contour.QuadContourSet to a GeoDataFrame
    Set an attribute `field_name` on each feature, according to `levels` values
    (`levels` must have the same number of features as the collection of contours)

    Parameters
    ----------
    collection_polygons: matplotlib.contour.QuadContourSet
        The result of a grid interpolation from matplotlib.
    levels: array-like
        The value to use as attributes for the constructed GeoDataFrame.
    field_name: string,
        The name of the field to be fill by `levels` variable (default: "levels")

    Returns
    -------
    gdf_contours: GeoDataFrame
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

    if len(data) == len(polygons):
        return GeoDataFrame(geometry=polygons,
                            data=data,
                            columns=[field_name])
    else:
        return GeoDataFrame(geometry=polygons)


def compute(knownpts, varname, span, beta, resolution,
            typefct='exponential', mask=None, longlat=False):
    if mask is None:
        tmp = (((knownpts.total_bounds[2] - knownpts.total_bounds[0])/10) +
               ((knownpts.total_bounds[3] - knownpts.total_bounds[1])/10)) / 2
        tmp = span if tmp < span else tmp
        bounds = knownpts.buffer(tmp).total_bounds
    else:
        bounds = mask.total_bounds

    unknownpts, shape = make_regular_points(bounds, resolution) if resolution \
        else make_regular_points_with_no_res(bounds)
    knwpts_coords = np.array([
        (g.coords.xy[0][0], g.coords.xy[1][0]) for g in knownpts.geometry.centroid])
    mat_dist = make_dist_mat(knwpts_coords, unknownpts, longlat=longlat)

    mat_dens = compute_interact_density(mat_dist, typefct, beta, span)
#    mat_opport = compute_opportunity(knownpts, mat_dens, varname)
#    pot = compute_potentials(mat_opport)
#    pot = compute_opportunity_potentials(knownpts[varname].values, mat_dens)
    pot = (knownpts[varname].values[:, np.newaxis] * mat_dens).sum(axis=0).round(8)

    return pot, unknownpts, shape
