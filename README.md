# smoomapy
----------


#### Requires:
 - GeoPandas
 - SciPy
 - Matplotlib

#### Usage example:
##### One-shot functionnality
```python
>>> result = quick_stewart('nuts3_data.geojson',
                           "pop1999",
                           span=65000,
                           beta=3,
                           resolution=48000,
                           mask='nuts3_data.geojson',
                           nb_class=10,
                           user_defined_breaks=None,
                           output="geodataframe")
```


##### Object-oriented API, allowing to easily redraw contours with new breaks values or new interpolation functionnality
```python
>>> StePot = SmoothStewart('nuts3_data.geojson', "pop1999",
                       span=65000, beta=3,
                       resolution=60000,
                       mask='nuts3_data.geojson')
>>> res = StePot.render(nb_class=8, func_grid="matplotlib",
                        disc_func="jenks", output="GeoDataFrame")
```

The long part of the computation is done during the initialization of `SmoothStewart` instance (i.e. actually computing potentials)
Some convenience methods allows to tweak and re-ewport the few last steps :

**_Allow to quickly redraw polygons with a new discretization
(or with new interpolation functionnality)_**
```python
>>> StePot.change_interp_grid_shape((164, 112))

>>> res = StePot.render(nb_class=6, func_grid="scipy",
                        disc_func="percentiles", output="GeoDataFrame")
```

**_Allow to set custom breaks values :_**
```python
>>> my_breaks = [0, 1697631, 3395263, 5092894, 6790526,
                 8488157, 10185789, 11883420, 13581052]

>>> res = StePot.render(nb_class=6, user_defined_breaks=my_breaks,
                        output="GeoDataFrame")
```

**_Some other smoothing methods can be used (experimental!) :_**
```python
>>> res = StePot.render(nb_class=12, func_grid="rbf", output="GeoDataFrame")
```