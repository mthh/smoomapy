#!/usr/bin/env python
# -*- coding: utf-8 -*-
import unittest
import numpy as np
import random
from geopandas import GeoDataFrame
from smoomapy import (
    quick_stewart, SmoothStewart,
    head_tail_breaks, maximal_breaks, get_opt_nb_class)
from smoomapy.helpers_classif import _chain


class TestSmoothStewart(unittest.TestCase):
    def setUp(self):
        pass

    def test_one_shot_stewart(self):
        # Exports correctly to `bytes`:
        res = quick_stewart(
            "misc/nuts3_data.geojson", "pop2008",
            span=65000, beta=2, resolution=60000, nb_class=8,
            mask="misc/nuts3_data.geojson")
        self.assertIsInstance(res, bytes)

        # Exports correctly to `GeoDataFrame`
        # and respects the choosen number of class:
        res = quick_stewart(
            "misc/nuts3_data.geojson", "pop2008",
            span=65000, beta=2, resolution=60000, nb_class=8,
            mask="misc/nuts3_data.geojson", output="GeoDataFrame")
        self.assertIsInstance(res, GeoDataFrame)
        self.assertEqual(len(res), 8)

        # Test that it works without specifying neither `nb_class` nor `resolution`:
        res = quick_stewart(
            "misc/nuts3_data.geojson", "pop2008",
            span=65000,
            beta=2,
            mask="misc/nuts3_data.geojson",
            output="GeoDataFrame")
        self.assertIsInstance(res, GeoDataFrame)

        # Test with user defined breaks values :
        my_breaks = [0, 1697631, 3395263, 5092894, 6790526,
                     8488157, 10185789, 11883420, 13581052]
        res = quick_stewart(
            "misc/nuts3_data.geojson",
            "pop2008",
            span=65000,
            beta=2,
            resolution=60000,
            user_defined_breaks=my_breaks,
            mask="misc/nuts3_data.geojson",
            output="GeoDataFrame")
        self.assertIsInstance(res, GeoDataFrame)
        self.assertEqual(len(res), 8)
        # Assert these break values were actually used :
        for wanted_break, obtained_break in zip(my_breaks[1:-1], res["max"][:-1]):
            self.assertAlmostEqual(wanted_break, obtained_break)

        # Test with user defined breaks values 
        # (the maximum value is volontarily low, a new class will be created to avoid making a hole)
        my_breaks = [0, 1697631, 3395263, 5092894, 6790526,
                     8488157, 10185789, 11883420, 12000000]
        res2 = quick_stewart(
            "misc/nuts3_data.geojson",
            "pop2008",
            span=65000,
            beta=2,
            resolution=60000,
            user_defined_breaks=my_breaks,
            mask="misc/nuts3_data.geojson",
            output="GeoDataFrame")
        self.assertIsInstance(res2, GeoDataFrame)

        # We can test that there is no hole by comparing the area of theses polygons
        # and the area of the previously computed resultat :
        self.assertAlmostEqual(res2.area.sum(), res.area.sum())
        # And by the fact that there is an extra class compared to the previous result :
        self.assertEqual(len(res2), 9)

    def test_object_stewart(self):
        # Test the OO approach for building smoothed map with stewart potentials
        StePot = SmoothStewart("misc/nuts3_data.geojson", "pop2008",
                               span=65000, beta=2, resolution=60000,
                               mask="misc/nuts3_data.geojson")

        # Test using percentiles :
        result = StePot.render(nb_class=10,
                               disc_func="percentiles",
                               output="geodataframe")
        self.assertIsInstance(result, GeoDataFrame)
        self.assertEqual(len(result), 10)

        # Test using an other griddata functionnality than the default one :
        result = StePot.render(nb_class=7,
                               disc_func="equal_interval",
                               func_grid="scipy-cubic",
                               output="geodataframe")
        self.assertIsInstance(result, GeoDataFrame)
        self.assertEqual(len(result), 7)

        # Test using somes already choosed break values :
        my_breaks = [0, 1697631, 3395263, 5092894, 6790526,
                     8488157, 10185789, 11883420, 13581052]
        result = StePot.render(
            nb_class=48,  # bogus values as `nb_class` and
            disc_func="foobar",  # ... disc_func should be overrided
            user_defined_breaks=my_breaks,  # ... by the `user_defined_breaks` params
            output="geodataframe")         # ... and this is what we are testing here

        self.assertIsInstance(result, GeoDataFrame)
        self.assertEqual(len(result), 8)
        # Assert these break values were actually used :
        for wanted_break, obtained_break in zip(my_breaks[1:-1], result["max"][:-1]):
            self.assertAlmostEqual(wanted_break, obtained_break)

        # Test again using another discretization method : "head tail breaks"
        # (should define automatically the number of class)
        result = StePot.render(nb_class=None,
                               disc_func="head_tail",
                               output="geodataframe")
        self.assertIsInstance(result, GeoDataFrame)

        # Test that the object has a nice representation :
        a = str(StePot)
        b = repr(StePot)
        self.assertEqual(a, b)
        self.assertIn("SmoothStewart - variable :", a)
        self.assertIn("{} features".format(len(StePot.gdf)), a)

    def test_object_stewart_two_var(self):
        # Test the OO approach with two variables :
        StePot = SmoothStewart("misc/nuts3_data.geojson", "gdppps2008",
                               span=65000, beta=2, resolution=48000,
                               variable_name2="pop2008",
                               mask="misc/nuts3_data.geojson")
        result = StePot.render(8, "equal_interval", output="Geodataframe")
        self.assertIsInstance(result, GeoDataFrame)
        self.assertEqual(len(result), 8)

    def test_distance_not_geo(self):
        # First whith one variable :
        StePot = SmoothStewart("misc/nuts3_data.geojson", "gdppps2008",
                               span=65000, beta=3, resolution=48000,
                               mask="misc/nuts3_data.geojson", distGeo=False)
        result = StePot.render(8, "equal_interval", output="Geodataframe")
        self.assertIsInstance(result, GeoDataFrame)
        self.assertEqual(len(result), 8)

        # Then with two variables :
        StePot = SmoothStewart("misc/nuts3_data.geojson", "gdppps2008",
                               span=65000, beta=2, resolution=48000,
                               variable_name2="pop2008",
                               mask="misc/nuts3_data.geojson", distGeo=False)
        result = StePot.render(8, "equal_interval", output="Geodataframe")
        self.assertIsInstance(result, GeoDataFrame)
        self.assertEqual(len(result), 8)

    def test_from_gdf_with_new_mask(self):
        gdf = GeoDataFrame.from_file("misc/nuts3_data.geojson")

        # Let's use pareto function for this one :
        StePot = SmoothStewart(gdf, "gdppps2008", typefct="pareto",
                               span=65000, beta=2.33, resolution=48000,
                               mask=None)
        result = StePot.render(6, output="Geodataframe")
        self.assertIsInstance(result, GeoDataFrame)
        self.assertEqual(len(result), 6)

        # Finally, use a mask (from a file) :
        result = StePot.render(5, output="Geodataframe",
                               new_mask="misc/nuts3_data.geojson")
        self.assertIsInstance(result, GeoDataFrame)
        self.assertEqual(StePot.use_mask, True)
        self.assertEqual(len(result), 5)

        # Or from a GeoDataFrame :
        gdf.geometry = gdf.geometry.buffer(100)

        result = StePot.render(6, output="Geodataframe",
                               new_mask=gdf)
        self.assertIsInstance(result, GeoDataFrame)
        self.assertEqual(StePot.use_mask, True)
        self.assertEqual(len(result), 6)

        # Nope, no mask :
        result = StePot.render(5, output="Geodataframe",
                               new_mask=None)
        self.assertIsInstance(result, GeoDataFrame)
        self.assertEqual(StePot.use_mask, False)
        self.assertEqual(len(result), 5)

        # Test that it skips the mask parameter if the layer provided as a mask
        # is not a Polygon/MultiPolygon layer :
        gdf_mask = gdf[1:50].copy()
        gdf_mask.geometry = gdf_mask.geometry.centroid
        result = StePot.render(5, output="Geodataframe",
                               new_mask=gdf_mask)
        self.assertIsInstance(result, GeoDataFrame)
        self.assertEqual(StePot.use_mask, False)
        self.assertEqual(len(result), 5)

    def test_input_with_missing_values(self):
        gdf = GeoDataFrame.from_file("misc/nuts3_data.geojson").to_crs({"init": "epsg:4326"})
        gdf.loc[12:18, "gdppps2008"] = np.NaN
        StePot = SmoothStewart(gdf, "gdppps2008",
                               span=65000, beta=2, resolution=48000,
                               mask=gdf)
        result = StePot.render(9, "equal_interval", output="Geodataframe")
        self.assertIsInstance(result, GeoDataFrame)
        self.assertEqual(len(result), 9)

    def test_wrong_dtype_missing_values(self):
        gdf = GeoDataFrame.from_file("misc/nuts3_data.geojson").to_crs({"init": "epsg:4326"})
        gdf.loc[12:18, "gdppps2008"] = np.NaN
        gdf.loc[25:35, "pop2008"] = np.NaN
        gdf.loc[0:len(gdf)-1, "pop2008"] = gdf["pop2008"].astype(str)
        StePot = SmoothStewart(gdf, "gdppps2008",
                               span=65000, beta=2, resolution=60000,
                               mask="misc/nuts3_data.geojson")
        result = StePot.render(9, "equal_interval", output="Geodataframe")
        self.assertIsInstance(result, GeoDataFrame)
        self.assertEqual(len(result), 9)

    def test_errors(self):
        # Test with a wrong interaction function name :
        with self.assertRaises(ValueError):
            StePot = SmoothStewart("misc/nuts3_data.geojson", "gdppps2008",
                                   span=65000, beta=2,
                                   typefct="abcdefg")

        StePot = SmoothStewart("misc/nuts3_data.geojson", "gdppps2008",
                               span=65000, beta=2, resolution=60000)

        # Test with a wrong discretization function name :
        with self.assertRaises(ValueError):
            StePot.render(9, "foo", output="Geodataframe")

        # Test with a wrong interpolation function name :
        with self.assertRaises(ValueError):
            StePot.render(9, "equal_interval", func_grid="foo")

        # Test using a layer without coordinate reference system :
        gdf = GeoDataFrame.from_file("misc/nuts3_data.geojson")
        gdf.crs = ""
        with self.assertRaises(Exception):
            SmoothStewart(gdf, "gdppps2008",
                          span=65000, beta=2, resolution=48000,
                          variable2="gdppps2008")

    def test_mod_shape_interpolation_grid(self):
        StePot = SmoothStewart("misc/nuts3_data.geojson", "pop2008",
                               span=65000, beta=2, resolution=75000,
                               mask="misc/nuts3_data.geojson")

        # First rendering :
        StePot.render(nb_class=8,
                      disc_func="percentiles",
                      output="geodataframe")

        # Change the shape of the interpolation grid
        # so the computed potential will be resampled to
        # draw the polygons :
        StePot.change_interp_grid_shape((40, 40))

        # Test using the default griddata function :
        result2 = StePot.render(nb_class=8,
                                disc_func="percentiles",
                                output="geodataframe")
        self.assertIsInstance(result2, GeoDataFrame)
        self.assertEqual(len(result2), 8)

        # ... using "nearest" method from scipy :
        result3 = StePot.render(nb_class=8,
                                disc_func="percentiles",
                                func_grid="scipy-nearest",
                                output="geodataframe")
        self.assertEqual(len(result3), 8)

        # ... using matplotlib mlab griddata function :
        result4 = StePot.render(nb_class=8,
                                disc_func="percentiles",
                                func_grid="matplotlib",
                                output="geodataframe")
        self.assertEqual(len(result4), 8)


class TestHelpers(unittest.TestCase):
    def setUp(self):
        self.li = [random.random() * 1000 for i in range(1200)]

    def test_head_tail_breaks(self):
        breaks = head_tail_breaks(self.li)
        self.assertIsInstance(breaks, list)

        breaks2 = head_tail_breaks(self.li, direction="head")
        self.assertIsInstance(breaks, list)
        self.assertAlmostEqual(breaks2, sorted(breaks2))
        self.assertAlmostEqual(breaks, breaks2)

        breaks3 = head_tail_breaks(self.li, direction="tail")
        self.assertIsInstance(breaks, list)
        self.assertAlmostEqual(breaks3, sorted(breaks3))

        with self.assertRaises(ValueError):
            head_tail_breaks(self.li, direction="nope")

    def test_maximal_breaks(self):
        breaks = maximal_breaks(self.li)
        self.assertIsInstance(breaks, list)

        breaks = maximal_breaks(self.li, k=6)
        self.assertIsInstance(breaks, list)
        self.assertEqual(len(breaks), 7)

    def test_get_opt_nb_class(self):
        nb_class = get_opt_nb_class(len(self.li))
        self.assertEqual(nb_class, 11)

    def test_chain_list(self):
        _list = [i for i in _chain([789, 45], [78, 96], [7878, 789, 36])]
        self.assertEqual(_list, [789, 45, 78, 96, 7878, 789, 36])


if __name__ == "__main__":
    unittest.main()
