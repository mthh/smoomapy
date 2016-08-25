# -*- coding: utf-8 -*-
import unittest
from geopandas import GeoDataFrame
from smoomapy import quick_stewart, SmoothStewart


class TestSmoothStewart(unittest.TestCase):
    def setUp(self):
        pass

    def test_one_shot_stewart(self):
        # Exports correctly to `bytes`:
        res = quick_stewart(
            "misc/nuts2_data.geojson", "pop2008",
            span=65000, beta=2, resolution=55000, nb_class=8,
            mask="misc/nuts2_data.geojson")
        self.assertIsInstance(res, bytes)

        # Exports correctly to `GeoDataFrame`
        # and respects the choosen number of class:
        res = quick_stewart(
            "misc/nuts2_data.geojson", "pop2008",
            span=65000, beta=2, resolution=55000, nb_class=8,
            mask="misc/nuts2_data.geojson", output="GeoDataFrame")
        self.assertIsInstance(res, GeoDataFrame)
        self.assertEqual(len(res), 8)

        # Test that it works without specifying neither `nb_class` nor `resolution`:
        res = quick_stewart(
            "misc/nuts2_data.geojson", "pop2008",
            span=65000,
            beta=2,
            mask="misc/nuts2_data.geojson",
            output="GeoDataFrame")
        self.assertIsInstance(res, GeoDataFrame)
        self.assertEqual(len(res), 8)

        # Test with user defined breaks values :
        my_breaks = [0, 1697631, 3395263, 5092894, 6790526,
                     8488157, 10185789, 11883420, 13581052]
        res = quick_stewart(
            "misc/nuts2_data.geojson", "pop2008",
            span=65000, beta=2, resolution=55000, user_defined_breaks=my_breaks,
            mask="misc/nuts2_data.geojson", output="GeoDataFrame")
        self.assertIsInstance(res, GeoDataFrame)
        self.assertEqual(len(res), 8)
        # Assert these break values were actually used :
        for wanted_break, obtained_break in zip(my_breaks[1:-1], res["max"][:-1]):
            self.assertAlmostEqual(wanted_break, obtained_break)

    def test_object_stewart(self):
        # Test the OO approach for building smoothed map with stewart potentials
        StePot = SmoothStewart("misc/nuts2_data.geojson", "pop2008",
                               span=65000, beta=2, resolution=48000,
                               mask="misc/nuts2_data.geojson")
        # Test using percentiles :
        result = StePot.render(nb_class=10,
                               disc_func="percentiles",
                               output="geodataframe")
        self.assertIsInstance(result, GeoDataFrame)
        self.assertEqual(len(result), 10)

        # Test using an other griddata functionnality than matplotlib :
        result = StePot.render(nb_class=7,
                               disc_func="equal_interval",
                               func_grid="scipy",
                               output="geodataframe")
        self.assertIsInstance(result, GeoDataFrame)
        self.assertEqual(len(result), 7)

        # Test using a ugly "self-made" discretization method
        result = StePot.render(nb_class=9,
                               disc_func="opt1",
                               output="geodataframe")
        self.assertIsInstance(result, GeoDataFrame)
        self.assertEqual(len(result), 9)

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


if __name__ == "__main__":
    unittest.main()
