smoomapy
========

Make smoothed maps in your python environnement
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

|Build Status Travis| |Build Status Appveyor| |Version| |Coveralls|

Takes an input of values located on Point features or on Polygon features (via their centroids)
and compute interpolated values on a grid using one of the following method:
**Stewart Potentials** or **Inverse Distance Weighting**.
*Grid resolution*, *distance function* (euclidian or haversine) and *projection* are configurable.

The resulting values are used to render a layer of contours, according to breaks values defined manually or using
a classification method such as *quantiles* or *Jenks natural breaks*.

This package had been partially developed for computing smoothed maps in **Magrit_**.
The "Stewart Potential" part is more or less a python port of *Stewart method*
from R **SpatialPositon_** package.

Input/output can be a path to a geographic layer (GeoJSON, shp, etc.) or a GeoDataFrame.

Requires:
^^^^^^^^^

-  Numpy
-  GeoPandas
-  Matplotlib

Documentation on the method :
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Please refer to https://github.com/Groupe-ElementR/SpatialPosition/ for documentation about **Stewarts potentials**.

Usage example:
~~~~~~~~~~~~~~

One-shot functionnality for Stewart potentials
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

    >>> result = quick_stewart('nuts3_data.geojson',
                               "pop1999",
                               span=65000,
                               beta=3,
                               resolution=48000,
                               mask='nuts3_data.geojson',
                               nb_class=10,
                               user_defined_breaks=None,
                               output="geojson")

One-shot functionnality for IDW
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

    >>> result = quick_idw('nuts3_data.geojson',
                           'pop1999',
                           power=1,
                           mask='nuts3_data.geojson',
                           nb_class=10,
                           user_defined_breaks=None,
                           output="geojson")

Object-oriented API for Stewart potentials
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

    >>> StePot = SmoothStewart('nuts3_data.geojson', "pop1999",
                               span=65000, beta=3,
                               resolution=60000,
                               mask='nuts3_data.geojson')
    >>> res = StePot.render(nb_class=8, disc_func="jenks",
                            output="GeoDataFrame")
    >>> res.plot(cmap="YlOrRd", linewidth=0.1)

.. figure:: https://raw.githubusercontent.com/mthh/smoomapy/master/misc/export_plot.png
   :alt: png_example


Object-oriented API for IDW
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

    >>> idw = SmoothIdw('nuts3_data.geojson', 'pop2008',
                        power=1, nb_pts=12000,
                        mask='nuts3_data.geojson')
    >>> res = idw.render(nb_class=8, disc_func="jenks",
                            output="GeoDataFrame")
    >>> res.plot(cmap="YlOrRd", linewidth=0.1)


Usage hints:
~~~~~~~~~~~~~~

The long part of the computation is done during actually interpolating values.
Using the Object-oriented API, you can compute new contours with the same interpolated values using the `render` method
(and using an other classification method within available ones: "equal\_interval", "prog\_geom", "jenks", "percentiles" and
"head-tail-breaks").

**Redraw polygons with a new classification method**

.. code:: python

    >>> res = StePot.render(nb_class=6,
                            disc_func="percentiles",
                            output="GeoDataFrame")

**Redraw contours using custom break values
(highly recommended after a first rendering or having take a look at the distibution):**

.. code:: python

    >>> my_breaks = [0, 1697631, 3395263, 5092894, 6790526,
                     8488157, 10185789, 11883420, 13581052]

    >>> res = StePot.render(nb_class=6, user_defined_breaks=my_breaks,
                            output="GeoDataFrame")

Installation:
~~~~~~~~~~~~~

From PyPI :
^^^^^^^^^^^

.. code:: shell

    $ pip install smoomapy


From github :
^^^^^^^^^^^^^

.. code:: shell

    $ git clone http://github.com/mthh/smoomapy.git
    $ cd smoomapy/
    $ python setup.py install

.. |Build Status Travis| image:: https://travis-ci.org/mthh/smoomapy.svg?branch=master
   :target: https://travis-ci.org/mthh/smoomapy

.. |Build Status Appveyor| image:: https://ci.appveyor.com/api/projects/status/tc7ynr2ejpp8a4c9/branch/master?svg=true
   :target: https://ci.appveyor.com/project/mthh/smoomapy/branch/master

.. |Version| image:: https://img.shields.io/pypi/v/smoomapy.svg
   :target: https://pypi.python.org/pypi/smoomapy

.. |Coveralls| image:: https://coveralls.io/repos/github/mthh/smoomapy/badge.svg?branch=master
   :target: https://coveralls.io/github/mthh/smoomapy?branch=master

.. _Magrit: http://magrit.cnrs.fr/

.. _SpatialPositon: https://github.com/Groupe-ElementR/SpatialPosition/
