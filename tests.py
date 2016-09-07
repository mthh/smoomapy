#!/usr/bin/env python
# -*- coding: utf-8 -*-
import unittest
import numpy as np
from geopandas import GeoDataFrame
from smoomapy import quick_stewart, SmoothStewart


class TestSmoothStewart(unittest.TestCase):
    def setUp(self):
        pass

    def test_one_shot_stewart(self):
        # Exports correctly to `bytes`:
        res = quick_stewart(
            "misc/nuts3_data.geojson", "pop2008",
            span=65000, beta=2, resolution=55000, nb_class=8,
            mask="misc/nuts3_data.geojson")
        self.assertIsInstance(res, bytes)

        # Exports correctly to `GeoDataFrame`
        # and respects the choosen number of class:
        res = quick_stewart(
            "misc/nuts3_data.geojson", "pop2008",
            span=65000, beta=2, resolution=55000, nb_class=8,
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
            "misc/nuts3_data.geojson", "pop2008",
            span=65000, beta=2, resolution=55000, user_defined_breaks=my_breaks,
            mask="misc/nuts3_data.geojson", output="GeoDataFrame")
        self.assertIsInstance(res, GeoDataFrame)
        self.assertEqual(len(res), 8)
        # Assert these break values were actually used :
        for wanted_break, obtained_break in zip(my_breaks[1:-1], res["max"][:-1]):
            self.assertAlmostEqual(wanted_break, obtained_break)

    def test_object_stewart(self):
        # Test the OO approach for building smoothed map with stewart potentials
        StePot = SmoothStewart("misc/nuts3_data.geojson", "pop2008",
                               span=65000, beta=2, resolution=48000,
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
        result = StePot.render(nb_class=48,  # bogus values as `nb_class` and
                               disc_func="foobar",  # ... disc_func should be overrided
                               user_defined_breaks=my_breaks,  # ... by the `user_defined_breaks` params
                               output="geodataframe")         # ... and this is what we are testing here
        self.assertIsInstance(result, GeoDataFrame)
        self.assertEqual(len(result), 8)
        # Assert these break values were actually used :
        for wanted_break, obtained_break in zip(my_breaks[1:-1], result["max"][:-1]):
            self.assertAlmostEqual(wanted_break, obtained_break)

        # Some tests on an other variables for testing another discretization method :
        StePot = SmoothStewart("misc/nuts3_data.geojson", "gdppps2008",
                               span=65000, beta=2, resolution=48000,
                               mask="misc/nuts3_data.geojson")
        # Using "head tail breaks" (should define automatically the number of class)
        result = StePot.render(nb_class=None,
                               disc_func="head_tail",
                               output="geodataframe")
        self.assertIsInstance(result, GeoDataFrame)


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
                               span=65000, beta=2, resolution=48000,
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

    def test_from_gdf(self):
        gdf = GeoDataFrame.from_file("misc/nuts3_data.geojson")
        StePot = SmoothStewart(gdf, "gdppps2008",
                               span=65000, beta=2, resolution=48000,
                               mask=gdf)
        result = StePot.render(9, "equal_interval", output="Geodataframe")
        self.assertIsInstance(result, GeoDataFrame)
        self.assertEqual(len(result), 9)

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
                               span=65000, beta=2, resolution=48000,
                               mask="misc/nuts3_data.geojson")
        result = StePot.render(9, "equal_interval", output="Geodataframe")
        self.assertIsInstance(result, GeoDataFrame)
        self.assertEqual(len(result), 9)

    def test_errors(self):
        StePot = SmoothStewart("misc/nuts3_data.geojson", "gdppps2008",
                               span=65000, beta=2, resolution=48000)
        with self.assertRaises(ValueError):
            StePot.render(9, "foo", output="Geodataframe")
        with self.assertRaises(ValueError):
            StePot.render(9, "equal_interval", func_grid="foo")

        gdf = GeoDataFrame.from_file("misc/nuts3_data.geojson")
        gdf.crs = ""
        with self.assertRaises(Exception):
            SmoothStewart(gdf, "gdppps2008",
                          span=65000, beta=2, resolution=48000,
                          variable2="gdppps2008")


if __name__ == "__main__":
    unittest.main()
