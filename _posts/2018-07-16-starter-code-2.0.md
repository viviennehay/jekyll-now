
---
title:  <img src="http://imgur.com/1ZcRyrc.png" style="float: left; margin: 20px; height: 55px"> World Values Analysis
date: 2017-06-16
published: true
---

<br> <br>



# Spatial Data Lab

_Authors: Matt Brems (DC)_


```python
# widget
```

## NYC Data Component
You should consult the [Geopandas Practice Notbook](geopandas-practice.ipynb) before diving into this lab.

In that notebook, you're introduced to the `GeoDataFrame` object from `geopandas`. A `GeoDataFrame` is just like a `DataFrame`, except it contains a `geometry` column that identifies each row as an object in space. A row can either represent a point in space (in which case the `geometry` column contains `Points`) or an area (in which case the `geometry` column contains `Polygons`). A `GeoDataFrame` can contain more than one column which contains spatial information, but only one column at a time can identify the unique geometry of an observation.

Here, we'll practice some of the same functionality and concepts.


```python
# basic stuff
import os
import pandas as pd
import numpy as np
from datetime import datetime
from urllib.request import urlretrieve
from zipfile import ZipFile
import pysal

# geo stuff
import geopandas as gpd
from shapely.geometry import Point
# from ipyleaflet import (Map,
#     Marker,
#     TileLayer, ImageOverlay,
#     Polyline, Polygon, Rectangle, Circle, CircleMarker,
#     GeoJSON,
#     DrawControl
# )

# plotting stuff
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('bmh')
plt.rcParams['figure.figsize'] = (10.0, 10.0)

# widget stuff
from ipywidgets import interact, HTML, FloatSlider
from IPython.display import clear_output, display

# progress stuff
from tqdm import tqdm_notebook, tqdm_pandas

# turn warnings off
import warnings
warnings.filterwarnings('ignore')
```


```python
# from the Geopandas practice notebook:

def get_nyc_shape_file(url, filename):

    # download file
    zipped = filename + '.zip'
    urlretrieve('https://data.cityofnewyork.us/api/geospatial/tqmj-j8zm?method=export&format=Shapefile', zipped)
    zipped = os.getcwd() + '/' + zipped

    # unzip file
    to_unzip = ZipFile(zipped, 'r')
    unzipped = os.getcwd() + '/' + filename + '_unzipped'
    to_unzip.extractall(unzipped)
    to_unzip.close()

    # get shape file
    for file in os.listdir(unzipped):
        if file.endswith(".shp"):
            shape_file = unzipped + '/' + file

    # return full file path
    return shape_file

# get shape file path
shape_file_url = 'https://data.cityofnewyork.us/api/geospatial/tqmj-j8zm?method=export&format=Shapefile'
shape_file_dir = 'nyc_boroughs'
file_path = get_nyc_shape_file(shape_file_url,shape_file_dir)

# read and view GeoDataFrame
gdf = gpd.GeoDataFrame.from_file(file_path)
gdf.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>boro_name</th>
      <th>boro_code</th>
      <th>shape_leng</th>
      <th>shape_area</th>
      <th>geometry</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Manhattan</td>
      <td>1.0</td>
      <td>361649.881587</td>
      <td>6.366006e+08</td>
      <td>(POLYGON ((-74.01092841268031 40.6844914725429...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Bronx</td>
      <td>2.0</td>
      <td>463464.521828</td>
      <td>1.186615e+09</td>
      <td>(POLYGON ((-73.89680883223774 40.7958084451597...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Staten Island</td>
      <td>5.0</td>
      <td>330432.867999</td>
      <td>1.623921e+09</td>
      <td>(POLYGON ((-74.05050806403247 40.5664220341608...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Brooklyn</td>
      <td>3.0</td>
      <td>739945.437431</td>
      <td>1.937567e+09</td>
      <td>(POLYGON ((-73.86706149472118 40.5820879767934...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Queens</td>
      <td>4.0</td>
      <td>895228.960360</td>
      <td>3.044772e+09</td>
      <td>(POLYGON ((-73.83668274106707 40.5949466970158...</td>
    </tr>
  </tbody>
</table>
</div>



#### To begin, return a `Series` containing the area of each NYC borough.

Does it match the area we are given? What units do you think these columns are in?

You will want to consult [the Geopandas docs](http://geopandas.org/reference.html) to familiarize yourself with the special attributes and methods of `GeoSeries` and `GeoDataFrame` objects.


```python
gdf['area_series'] = gdf["geometry"].area
```


```python
gdf["area_series_ratio"] = gdf.shape_area/gdf.area_series
```


```python
gdf.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>boro_name</th>
      <th>boro_code</th>
      <th>shape_leng</th>
      <th>shape_area</th>
      <th>geometry</th>
      <th>area_series</th>
      <th>area_series_ratio</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Manhattan</td>
      <td>1.0</td>
      <td>361649.881587</td>
      <td>6.366006e+08</td>
      <td>(POLYGON ((-74.01092841268031 40.6844914725429...</td>
      <td>0.006309</td>
      <td>1.009057e+11</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Bronx</td>
      <td>2.0</td>
      <td>463464.521828</td>
      <td>1.186615e+09</td>
      <td>(POLYGON ((-73.89680883223774 40.7958084451597...</td>
      <td>0.011773</td>
      <td>1.007926e+11</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Staten Island</td>
      <td>5.0</td>
      <td>330432.867999</td>
      <td>1.623921e+09</td>
      <td>(POLYGON ((-74.05050806403247 40.5664220341608...</td>
      <td>0.016047</td>
      <td>1.012008e+11</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Brooklyn</td>
      <td>3.0</td>
      <td>739945.437431</td>
      <td>1.937567e+09</td>
      <td>(POLYGON ((-73.86706149472118 40.5820879767934...</td>
      <td>0.019164</td>
      <td>1.011048e+11</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Queens</td>
      <td>4.0</td>
      <td>895228.960360</td>
      <td>3.044772e+09</td>
      <td>(POLYGON ((-73.83668274106707 40.5949466970158...</td>
      <td>0.030143</td>
      <td>1.010102e+11</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Our "area_series" is similar to the "shape_area", but 10**11 smaller
# The area of Manhattan is 23 sq miles = 6.36*10**8 sq ft. So the "shape_area" column is likely in sq. ft.
# The geometry coords are in lat/long, so the "area_series" column must be in those units. 
# The 10**11 is the conversion from those units to sq ft and our specific location on the globe
```

#### Add a new column to the dataset containing the centroid of each borough.

What type of object is this? What type of object does it contain?
Can we make this the `geometry` column for this dataset?


```python
gdf['centroid'] = gdf["geometry"].centroid
```


```python
type(gdf['centroid'])
```




    pandas.core.series.Series




```python
f'The column is a {type(gdf["centroid"])}. It contains {type(gdf["centroid"][1])}'
```




    "The column is a <class 'pandas.core.series.Series'>. It contains <class 'shapely.geometry.point.Point'>"




```python
# Yes, we could make this the geometry column, as the geometry can either be a point or a shape
```

#### Now, plot the NYC boroughs, the convex hull for each borough, and the envelope for each borough.

Hint: You can call `.plot` on a `GeoDataFrame`.


```python
x_list = list(gdf["geometry"].centroid.apply(lambda p: p.x))
y_list = list(gdf["geometry"].centroid.apply(lambda p: p.y))
gdf["centroid coords"] = list(zip(x_list, y_list))
gdf["centroid coords"][0][0]
```




    -73.96716968771112




```python
x_index = 0
y_index = 1
offset = 0.01

fig, ax = plt.subplots()
gdf["geometry"].plot(ax = ax, cmap = "Blues")
gdf["geometry"].centroid.plot(ax = ax, c = 'b')
for i, label in enumerate(gdf["boro_name"]):
    ax.text(gdf["centroid coords"][i][x_index]+ offset, gdf["centroid coords"][i][y_index], label, 
            size=8, rotation=30.,
             ha="left", va="bottom",
#              bbox=dict(boxstyle="round",
#                    ec=(1., 0.5, 0.5),
#                    fc=(1., 0.8, 0.8),
#                    )
             )
```


![png](/images/starter-code-2.0_files/starter-code-2.0_17_0.png)



```python
x_index = 0
y_index = 1
offset = 0.01

fig, ax = plt.subplots()
gdf["geometry"].convex_hull.plot(ax = ax, cmap = "Blues")
gdf["geometry"].centroid.plot(ax = ax, c = 'b')
for i, label in enumerate(gdf["boro_name"]):
    ax.text(gdf["centroid coords"][i][x_index], gdf["centroid coords"][i][y_index]+offset, label, 
            size=8, rotation=30.,
             ha="right", va="center",
#              bbox=dict(boxstyle="round",
#                    ec=(1., 0.5, 0.5),
#                    fc=(1., 0.8, 0.8),
#                    )
             )
```


![png](/images/starter-code-2.0_files/starter-code-2.0_18_0.png)



```python
x_index = 0
y_index = 1
offset = 0.01

fig, ax = plt.subplots()
gdf["geometry"].envelope.plot(ax = ax, cmap = "Blues")
gdf["geometry"].centroid.plot(ax = ax, c = 'b')
for i, label in enumerate(gdf["boro_name"]):
    ax.text(gdf["centroid coords"][i][x_index], gdf["centroid coords"][i][y_index]+offset, label, 
            size=8, rotation=30.,
             ha="right", va="center",
#              bbox=dict(boxstyle="round",
#                    ec=(1., 0.5, 0.5),
#                    fc=(1., 0.8, 0.8),
#                    )
             )
```


![png](/images/starter-code-2.0_files/starter-code-2.0_19_0.png)


#### Bonus: Plot the centroid of each borough on the map of each borough


```python
# done
```

#### Generate 10,000 samples uniformly across the NYC map. 

Note, you're generating both a random X and a random Y in order to get a location on the NYC map, much like how you might estimate $\pi$ using Monte Carlo simulations.

Plot these points over the map of NYC.


```python
# from the maps, we can see that the axes range from -74.3 to -73.7 on the x-axis and 40.4 to 50.0 on the y-axis
np.random.seed(42)
rand_X = list(np.random.uniform(-74.3, -73.7, 10000))
rand_Y = list(np.random.uniform(40.4, 41.0, 10000))
random_points = list(zip(rand_X, rand_Y))
random_points[0:5]
```




    [(-74.07527592869158, 40.624184491080015),
     (-73.72957141615406, 40.59974725773884),
     (-73.86080363491315, 40.505692347501714),
     (-73.94080490948178, 40.76436000206089),
     (-74.20638881573454, 40.68597449630518)]



#### Place points within boroughs
A common geosptial task is to check whether a given point lies inside or outside of a certain area. In order to ease that calculation, convex hulls and envelopes are often used as approximations of the true shape of geographical areas.

In this part, we'll check which (if any) each borough our simulated points fall into:

- Whether or not each sample falls in the true geographic boroughs.
- Whether or not each sample falls in the convex hulls of the boroughs.
- Whether or not each sample falls in the envelopes of the boroughs.

We'll need to use the `Point` object that we imported from `shapely` and the `.contains` method from Geopandas.

At each step, use the `%%timeit` [magic command](https://ipython.readthedocs.io/en/stable/interactive/magics.html) to measure how long this process takes.

Report these numbers, as well as how much more efficient (percentage-wise) envelopes and convex hulls are relative to the true geographies.


```python
# first let's test the method using our centroids

gdf["geometry"].envelope.contains(gdf["centroid"][3])
# this is working correctly - the centroid of Brooklyn is in the envelopes for both Brooklyn and Queens
```




    0    False
    1    False
    2    False
    3     True
    4     True
    dtype: bool




```python
gdf.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>boro_name</th>
      <th>boro_code</th>
      <th>shape_leng</th>
      <th>shape_area</th>
      <th>geometry</th>
      <th>area_series</th>
      <th>area_series_ratio</th>
      <th>centroid</th>
      <th>centroid coords</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Manhattan</td>
      <td>1.0</td>
      <td>361649.881587</td>
      <td>6.366006e+08</td>
      <td>(POLYGON ((-74.01092841268031 40.6844914725429...</td>
      <td>0.006309</td>
      <td>1.009057e+11</td>
      <td>POINT (-73.96716968771112 40.77726401016272)</td>
      <td>(-73.96716968771112, 40.77726401016272)</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Bronx</td>
      <td>2.0</td>
      <td>463464.521828</td>
      <td>1.186615e+09</td>
      <td>(POLYGON ((-73.89680883223774 40.7958084451597...</td>
      <td>0.011773</td>
      <td>1.007926e+11</td>
      <td>POINT (-73.86654163194733 40.85262967859273)</td>
      <td>(-73.86654163194733, 40.852629678592734)</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Staten Island</td>
      <td>5.0</td>
      <td>330432.867999</td>
      <td>1.623921e+09</td>
      <td>(POLYGON ((-74.05050806403247 40.5664220341608...</td>
      <td>0.016047</td>
      <td>1.012008e+11</td>
      <td>POINT (-74.15336979029414 40.58085472747923)</td>
      <td>(-74.15336979029414, 40.58085472747923)</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Brooklyn</td>
      <td>3.0</td>
      <td>739945.437431</td>
      <td>1.937567e+09</td>
      <td>(POLYGON ((-73.86706149472118 40.5820879767934...</td>
      <td>0.019164</td>
      <td>1.011048e+11</td>
      <td>POINT (-73.94767501523242 40.64473132725909)</td>
      <td>(-73.94767501523242, 40.644731327259095)</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Queens</td>
      <td>4.0</td>
      <td>895228.960360</td>
      <td>3.044772e+09</td>
      <td>(POLYGON ((-73.83668274106707 40.5949466970158...</td>
      <td>0.030143</td>
      <td>1.010102e+11</td>
      <td>POINT (-73.81849453624231 40.70760815187309)</td>
      <td>(-73.81849453624231, 40.70760815187309)</td>
    </tr>
  </tbody>
</table>
</div>




```python
# now turn our random points into geopandas points
random_points = pd.DataFrame(random_points).T.apply(Point)
```


```python
type(random_points)
```




    pandas.core.series.Series




```python
type(random_points)
```




    pandas.core.series.Series




```python
# in order to plot the simulated data on a map, we need to turn this into a geopandas dataframe
# for consistency, let's make sure this is the same coordinate reference system as the original geodataframe
gdf.crs
```




    {'init': 'epsg:4326'}




```python
random_points.crs = {'init' :'epsg:4326'}
```


```python
# check types
```


```python
type(gdf), type(random_points)
```




    (geopandas.geodataframe.GeoDataFrame, pandas.core.series.Series)




```python
# We also see that random_points is a series. In order to join, both should be geodataframes
# We set the crs in the random_points gdf equal to the crs in the gdf above, and the geometry as the (untitled) 0 column)
```


```python
random_points_gdf = gpd.GeoDataFrame(random_points, crs= {'init' :'epsg:4326'}, columns = ['points'], geometry = 'points')
```


```python
random_points_gdf.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>points</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>POINT (-74.07527592869158 40.62418449108002)</td>
    </tr>
    <tr>
      <th>1</th>
      <td>POINT (-73.72957141615406 40.59974725773884)</td>
    </tr>
    <tr>
      <th>2</th>
      <td>POINT (-73.86080363491315 40.50569234750171)</td>
    </tr>
    <tr>
      <th>3</th>
      <td>POINT (-73.94080490948178 40.76436000206089)</td>
    </tr>
    <tr>
      <th>4</th>
      <td>POINT (-74.20638881573454 40.68597449630518)</td>
    </tr>
  </tbody>
</table>
</div>




```python
gdf.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>boro_name</th>
      <th>boro_code</th>
      <th>shape_leng</th>
      <th>shape_area</th>
      <th>geometry</th>
      <th>area_series</th>
      <th>area_series_ratio</th>
      <th>centroid</th>
      <th>centroid coords</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Manhattan</td>
      <td>1.0</td>
      <td>361649.881587</td>
      <td>6.366006e+08</td>
      <td>(POLYGON ((-74.01092841268031 40.6844914725429...</td>
      <td>0.006309</td>
      <td>1.009057e+11</td>
      <td>POINT (-73.96716968771112 40.77726401016272)</td>
      <td>(-73.96716968771112, 40.77726401016272)</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Bronx</td>
      <td>2.0</td>
      <td>463464.521828</td>
      <td>1.186615e+09</td>
      <td>(POLYGON ((-73.89680883223774 40.7958084451597...</td>
      <td>0.011773</td>
      <td>1.007926e+11</td>
      <td>POINT (-73.86654163194733 40.85262967859273)</td>
      <td>(-73.86654163194733, 40.852629678592734)</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Staten Island</td>
      <td>5.0</td>
      <td>330432.867999</td>
      <td>1.623921e+09</td>
      <td>(POLYGON ((-74.05050806403247 40.5664220341608...</td>
      <td>0.016047</td>
      <td>1.012008e+11</td>
      <td>POINT (-74.15336979029414 40.58085472747923)</td>
      <td>(-74.15336979029414, 40.58085472747923)</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Brooklyn</td>
      <td>3.0</td>
      <td>739945.437431</td>
      <td>1.937567e+09</td>
      <td>(POLYGON ((-73.86706149472118 40.5820879767934...</td>
      <td>0.019164</td>
      <td>1.011048e+11</td>
      <td>POINT (-73.94767501523242 40.64473132725909)</td>
      <td>(-73.94767501523242, 40.644731327259095)</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Queens</td>
      <td>4.0</td>
      <td>895228.960360</td>
      <td>3.044772e+09</td>
      <td>(POLYGON ((-73.83668274106707 40.5949466970158...</td>
      <td>0.030143</td>
      <td>1.010102e+11</td>
      <td>POINT (-73.81849453624231 40.70760815187309)</td>
      <td>(-73.81849453624231, 40.70760815187309)</td>
    </tr>
  </tbody>
</table>
</div>




```python
x_index = 0
y_index = 1
offset = 0.005

fig, ax = plt.subplots(figsize = (8,5))
plt.title('\n Simulated taxi pickups in NYC \n ')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
gdf["geometry"].plot(ax = ax, cmap = "cool")
random_points_gdf['points'].plot(ax = ax, markersize = 0.3, c = 'b', label = 'Simulated points')
gdf["geometry"].centroid.plot(ax = ax, c = 'b')
for i, label in enumerate(gdf["boro_name"]):
    ax.text(gdf["centroid coords"][i][x_index]+ offset, gdf["centroid coords"][i][y_index]+offset, 
            label, 
            size=8, rotation=45.,
            ha="left", va="bottom",
            bbox=dict(facecolor = 'white', edgecolor='none')
             )
plt.legend(facecolor = 'white', framealpha = 1);
```


![png](/images/starter-code-2.0_files/starter-code-2.0_38_0.png)


** Shapes **


```python
%%timeit 
# timing cell magic - just on 20 points to reduce time, since this cycles throuhg 3x

shape_df = pd.DataFrame([])

for i in range(0, 20):
    row = gdf["geometry"].contains(random_points[i])
    shape_df = shape_df.append(row, ignore_index=True)
```

    294 ms ± 22.1 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)



```python
# now test whether each point is contained in the shape

shape_df = pd.DataFrame([])

for i in range(0, len(random_points)):
    row = gdf["geometry"].contains(random_points[i])
    shape_df = shape_df.append(row, ignore_index=True)

model = "true"
true_col_list = [f'{model}_{boro}' for boro in gdf["boro_name"]]
shape_df.columns = true_col_list 
shape_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>true_Manhattan</th>
      <th>true_Bronx</th>
      <th>true_Staten Island</th>
      <th>true_Brooklyn</th>
      <th>true_Queens</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



** Convex hull **


```python
%%timeit # timing cell magic

hull_df = pd.DataFrame([])

for i in range(0, 20):
    row = gdf["geometry"].convex_hull.contains(random_points[i])
    hull_df = hull_df.append(row, ignore_index=True)
```

    2.38 s ± 8.12 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)



```python
# now test whether each point is contained in the shape

hull_df = pd.DataFrame([])

for i in range(0, len(random_points)):
    row = gdf["geometry"].convex_hull.contains(random_points[i])
    hull_df = hull_df.append(row, ignore_index=True)

model = "hull"
hull_col_list = [f'{model}_{boro}' for boro in gdf["boro_name"]]
hull_df.columns = hull_col_list 
hull_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>hull_Manhattan</th>
      <th>hull_Bronx</th>
      <th>hull_Staten Island</th>
      <th>hull_Brooklyn</th>
      <th>hull_Queens</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



** Envelope **


```python
%%timeit # timing cell magic

envelope_df = pd.DataFrame([])

for i in range(0, 20):
    row = gdf["geometry"].envelope.contains(random_points[i])
    envelope_df = envelope_df.append(row, ignore_index=True)
```

    51.9 ms ± 907 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)



```python
# now test whether each point is contained in the shape

envelope_df = pd.DataFrame([])

for i in range(0, len(random_points)):
    row = gdf["geometry"].envelope.contains(random_points[i])
    envelope_df = envelope_df.append(row, ignore_index=True)

model = "envelope"
envelope_col_list = [f'{model}_{boro}' for boro in gdf["boro_name"]]
envelope_df.columns = envelope_col_list 
envelope_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>envelope_Manhattan</th>
      <th>envelope_Bronx</th>
      <th>envelope_Staten Island</th>
      <th>envelope_Brooklyn</th>
      <th>envelope_Queens</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



#### Generate metrics.  Summarize findings.

Obviously there's a trade-off here. Check how many samples lie in the actual geographies, the convex hulls, and the envelopes.

Report the following:

- A confusion matrix comparing convex hulls and actual geographies. (i.e. actual geographies are the true counts; convex hulls are predicted counts)
- A confusion matrix comparing envelopes and actual geographies.
- The accuracy and sensitivity from each of the confusion matrices above. You should report a sensitivity value for each borough.
- A paragraph summarizing your findings.


```python
true_v_hull_df = pd.concat((shape_df, hull_df), axis = 1)
```


```python
true_v_hull_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>true_Manhattan</th>
      <th>true_Bronx</th>
      <th>true_Staten Island</th>
      <th>true_Brooklyn</th>
      <th>true_Queens</th>
      <th>hull_Manhattan</th>
      <th>hull_Bronx</th>
      <th>hull_Staten Island</th>
      <th>hull_Brooklyn</th>
      <th>hull_Queens</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
model = "true"
true_col_list = [f'{model}_{boro}' for boro in gdf["boro_name"]]
```


```python
model = "hull"
hull_col_list = [f'{model}_{boro}' for boro in gdf["boro_name"]]
```


```python
# Method 1
# true_v_hull_df["sum_true"] = true_v_hull_df[true_col_list].sum(axis = 1)
# true_v_hull_df["true_not_in_boro"] = [1 if true_v_hull_df["sum_true"].iloc[i] == 0 else 0 for i in true_v_hull_df.index]
```


```python
# Method 2
# true_v_hull_df["true_not_in_boro"] = [1 if true_v_hull_df[true_col_list].sum(axis = 1).iloc[i] == 0 else 0 for i in true_v_hull_df.index]
```


```python
# Final method
true_v_hull_df["true_not_in_boro"] = abs(true_v_hull_df[true_col_list].sum(axis = 1)-1)
true_v_hull_df["hull_not_in_boro"] = [1 if true_v_hull_df[hull_col_list].sum(axis = 1).iloc[i] == 0 else 0 for i in true_v_hull_df.index]
```


```python
true_v_hull_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>true_Manhattan</th>
      <th>true_Bronx</th>
      <th>true_Staten Island</th>
      <th>true_Brooklyn</th>
      <th>true_Queens</th>
      <th>hull_Manhattan</th>
      <th>hull_Bronx</th>
      <th>hull_Staten Island</th>
      <th>hull_Brooklyn</th>
      <th>hull_Queens</th>
      <th>true_not_in_boro</th>
      <th>hull_not_in_boro</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
true_v_hull_df = true_v_hull_df.astype(int)
```


```python
true_col_list.append("true_not_in_boro")
hull_col_list.append("hull_not_in_boro")
```


```python
hull_confusion_matrix = pd.DataFrame(index = true_col_list, 
                                     columns = hull_col_list)
hull_confusion_matrix.fillna(0, inplace=True)
```


```python
hull_confusion_matrix
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>hull_Manhattan</th>
      <th>hull_Bronx</th>
      <th>hull_Staten Island</th>
      <th>hull_Brooklyn</th>
      <th>hull_Queens</th>
      <th>hull_not_in_boro</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>true_Manhattan</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>true_Bronx</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>true_Staten Island</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>true_Brooklyn</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>true_Queens</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>true_not_in_boro</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
true_v_hull_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>true_Manhattan</th>
      <th>true_Bronx</th>
      <th>true_Staten Island</th>
      <th>true_Brooklyn</th>
      <th>true_Queens</th>
      <th>hull_Manhattan</th>
      <th>hull_Bronx</th>
      <th>hull_Staten Island</th>
      <th>hull_Brooklyn</th>
      <th>hull_Queens</th>
      <th>true_not_in_boro</th>
      <th>hull_not_in_boro</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
%%timeit

# loop through each true/ estimated combination
for index in true_col_list:
    for column in hull_col_list:
# count number of occurences
        i = 0
        for row in true_v_hull_df.index:
            if true_v_hull_df[index].iloc[row] == 1 and true_v_hull_df[column].iloc[row] == 1:
                i+=1
# update confusion matrix
        hull_confusion_matrix.loc[index,column] = i
```

    15.4 s ± 217 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)



```python
%%timeit

# loop through each true/ estimated combination
for index in true_col_list:
    for column in hull_col_list:
# count number of occurences
        for row in true_v_hull_df.index:
            i = np.dot(true_v_hull_df[index], true_v_hull_df[column])
# update confusion matrix
        hull_confusion_matrix.loc[index,column] = i
```

    25.4 s ± 150 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)



```python
%%timeit

# loop through each true/ estimated combination
for index in true_col_list:
    for column in hull_col_list:
# count number of occurences
        for row in true_v_hull_df.index:
            i = np.dot(true_v_hull_df[index], true_v_hull_df[column])
# update confusion matrix
        hull_confusion_matrix.loc[index,column] = i
```

    27.2 s ± 3.08 s per loop (mean ± std. dev. of 7 runs, 1 loop each)



```python
hull_confusion_matrix
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>hull_Manhattan</th>
      <th>hull_Bronx</th>
      <th>hull_Staten Island</th>
      <th>hull_Brooklyn</th>
      <th>hull_Queens</th>
      <th>hull_not_in_boro</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>true_Manhattan</th>
      <td>163</td>
      <td>9</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>true_Bronx</th>
      <td>32</td>
      <td>328</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
      <td>0</td>
    </tr>
    <tr>
      <th>true_Staten Island</th>
      <td>0</td>
      <td>0</td>
      <td>428</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>true_Brooklyn</th>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>546</td>
      <td>282</td>
      <td>0</td>
    </tr>
    <tr>
      <th>true_Queens</th>
      <td>18</td>
      <td>2</td>
      <td>0</td>
      <td>35</td>
      <td>829</td>
      <td>0</td>
    </tr>
    <tr>
      <th>true_not_in_boro</th>
      <td>93</td>
      <td>121</td>
      <td>114</td>
      <td>102</td>
      <td>293</td>
      <td>7068</td>
    </tr>
  </tbody>
</table>
</div>




```python
# accuracy = all correct / all
accuracy = np.diagonal(hull_confusion_matrix).sum() / hull_confusion_matrix.sum().sum()
print(f'accuracy = {accuracy*100:.0f}%')
```

    accuracy = 89%



```python
# sensitvity = true positives / all positives

for boro in gdf['boro_name']:
    boro_sens = hull_confusion_matrix.loc[f'true_{boro}', f'hull_{boro}']/hull_confusion_matrix[f'hull_{boro}'].sum()
    print(f'The sensitivity for {boro} is {boro_sens*100:.0f}%')
```

    The sensitivity for Manhattan is 52%
    The sensitivity for Bronx is 71%
    The sensitivity for Staten Island is 79%
    The sensitivity for Brooklyn is 80%
    The sensitivity for Queens is 59%



```python
# specificity = true negatives / all negatives
# this will be 100% as the convex hull is a superset of the shape itself.
```

Our overall accuracy for the convex hull is 89%, which is quite high. However, it actually takes *longer* to determine if a point is in the convex hull vs the original shape (2s v. 200ms for a loop of 20). The sensitivity is highest for Brooklyn and Staten island, and lowest for Manhattan and Queens as their convex hulls overlap with other boroughs.

We would see a similar pattern for envelope, though sensitivity tends to be lower because the envelope is a rougher approximation.

#### Perform a spatial join using your simulated data

You should consider the [Geopandas docs](http://geopandas.readthedocs.io/en/latest/reference/geopandas.sjoin.html).

Hint: You must use two `GeoDataFrame`s
Hint: Use `crs= {'init' :'epsg:4326'}`

##### First, use `sjoin` to label each simulated point according to its corresponding borough
This should give the same results as above, when you used `.contains` to check and see which borough each point belonged to.


```python
spatial_join_df = gpd.sjoin(random_points_gdf, gdf, how = 'left', op = 'within')
```


```python
spatial_join_df.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>points</th>
      <th>index_right</th>
      <th>boro_name</th>
      <th>boro_code</th>
      <th>shape_leng</th>
      <th>shape_area</th>
      <th>area_series</th>
      <th>area_series_ratio</th>
      <th>centroid</th>
      <th>centroid coords</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>POINT (-74.07527592869158 40.62418449108002)</td>
      <td>2.0</td>
      <td>Staten Island</td>
      <td>5.0</td>
      <td>330432.867999</td>
      <td>1.623921e+09</td>
      <td>0.016047</td>
      <td>1.012008e+11</td>
      <td>POINT (-74.15336979029414 40.58085472747923)</td>
      <td>(-74.15336979029414, 40.58085472747923)</td>
    </tr>
    <tr>
      <th>1</th>
      <td>POINT (-73.72957141615406 40.59974725773884)</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>POINT (-73.86080363491315 40.50569234750171)</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>POINT (-73.94080490948178 40.76436000206089)</td>
      <td>4.0</td>
      <td>Queens</td>
      <td>4.0</td>
      <td>895228.960360</td>
      <td>3.044772e+09</td>
      <td>0.030143</td>
      <td>1.010102e+11</td>
      <td>POINT (-73.81849453624231 40.70760815187309)</td>
      <td>(-73.81849453624231, 40.70760815187309)</td>
    </tr>
    <tr>
      <th>4</th>
      <td>POINT (-74.20638881573454 40.68597449630518)</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>5</th>
      <td>POINT (-74.20640328779828 40.9194205953944)</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>6</th>
      <td>POINT (-74.26514983269908 40.41926574828923)</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>7</th>
      <td>POINT (-73.78029431253503 40.78632075653307)</td>
      <td>4.0</td>
      <td>Queens</td>
      <td>4.0</td>
      <td>895228.960360</td>
      <td>3.044772e+09</td>
      <td>0.030143</td>
      <td>1.010102e+11</td>
      <td>POINT (-73.81849453624231 40.70760815187309)</td>
      <td>(-73.81849453624231, 40.70760815187309)</td>
    </tr>
    <tr>
      <th>8</th>
      <td>POINT (-73.93933099295407 40.85776932710694)</td>
      <td>0.0</td>
      <td>Manhattan</td>
      <td>1.0</td>
      <td>361649.881587</td>
      <td>6.366006e+08</td>
      <td>0.006309</td>
      <td>1.009057e+11</td>
      <td>POINT (-73.96716968771112 40.77726401016272)</td>
      <td>(-73.96716968771112, 40.77726401016272)</td>
    </tr>
    <tr>
      <th>9</th>
      <td>POINT (-73.87515645332238 40.85569194151299)</td>
      <td>1.0</td>
      <td>Bronx</td>
      <td>2.0</td>
      <td>463464.521828</td>
      <td>1.186615e+09</td>
      <td>0.011773</td>
      <td>1.007926e+11</td>
      <td>POINT (-73.86654163194733 40.85262967859273)</td>
      <td>(-73.86654163194733, 40.852629678592734)</td>
    </tr>
  </tbody>
</table>
</div>



##### Bonus: Use `sjoin` to count the number of points in each borough.


```python
# Counting points in each boro using spatial join
spatial_join_df["boro_name"].value_counts()
```




    Queens           829
    Brooklyn         546
    Staten Island    428
    Bronx            328
    Manhattan        163
    Name: boro_name, dtype: int64




```python
# Confirming vs our shape_df
shape_df.sum()
```




    true_Manhattan        163.0
    true_Bronx            328.0
    true_Staten Island    428.0
    true_Brooklyn         546.0
    true_Queens           829.0
    dtype: float64



#### Generate a map of NYC with each borough shaded based on the number of pick-ups that occur in each borough.


```python
## This will take awhile! Check out the data dictionary in the meantime: 
## http://www.nyc.gov/html/tlc/downloads/pdf/data_dictionary_trip_records_yellow.pdf

taxi = pd.read_csv("https://s3.amazonaws.com/nyc-tlc/trip+data/yellow_tripdata_2015-09.csv")
```


```python
taxi.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>VendorID</th>
      <th>tpep_pickup_datetime</th>
      <th>tpep_dropoff_datetime</th>
      <th>passenger_count</th>
      <th>trip_distance</th>
      <th>pickup_longitude</th>
      <th>pickup_latitude</th>
      <th>RatecodeID</th>
      <th>store_and_fwd_flag</th>
      <th>dropoff_longitude</th>
      <th>dropoff_latitude</th>
      <th>payment_type</th>
      <th>fare_amount</th>
      <th>extra</th>
      <th>mta_tax</th>
      <th>tip_amount</th>
      <th>tolls_amount</th>
      <th>improvement_surcharge</th>
      <th>total_amount</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>2015-09-01 00:05:55</td>
      <td>2015-09-01 00:31:02</td>
      <td>1</td>
      <td>17.45</td>
      <td>-73.791351</td>
      <td>40.646690</td>
      <td>1</td>
      <td>N</td>
      <td>-73.857437</td>
      <td>40.848263</td>
      <td>1</td>
      <td>47.5</td>
      <td>0.5</td>
      <td>0.5</td>
      <td>5.00</td>
      <td>5.54</td>
      <td>0.3</td>
      <td>59.34</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>2015-09-01 00:05:56</td>
      <td>2015-09-01 00:07:42</td>
      <td>1</td>
      <td>0.40</td>
      <td>-73.978935</td>
      <td>40.752853</td>
      <td>1</td>
      <td>N</td>
      <td>-73.986061</td>
      <td>40.755398</td>
      <td>2</td>
      <td>3.5</td>
      <td>0.5</td>
      <td>0.5</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.3</td>
      <td>4.80</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>2015-09-01 00:05:57</td>
      <td>2015-09-01 00:16:48</td>
      <td>1</td>
      <td>1.50</td>
      <td>-73.990891</td>
      <td>40.723972</td>
      <td>1</td>
      <td>N</td>
      <td>-74.009560</td>
      <td>40.728916</td>
      <td>2</td>
      <td>9.0</td>
      <td>0.5</td>
      <td>0.5</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.3</td>
      <td>10.30</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>2015-09-01 00:05:57</td>
      <td>2015-09-01 00:05:57</td>
      <td>1</td>
      <td>0.00</td>
      <td>-73.932655</td>
      <td>40.803768</td>
      <td>1</td>
      <td>N</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>2</td>
      <td>4.0</td>
      <td>0.5</td>
      <td>0.5</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.3</td>
      <td>5.30</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>2015-09-01 00:05:57</td>
      <td>2015-09-01 00:30:32</td>
      <td>1</td>
      <td>7.50</td>
      <td>-73.987778</td>
      <td>40.738194</td>
      <td>1</td>
      <td>N</td>
      <td>-73.944756</td>
      <td>40.828167</td>
      <td>1</td>
      <td>23.5</td>
      <td>0.5</td>
      <td>0.5</td>
      <td>4.95</td>
      <td>0.00</td>
      <td>0.3</td>
      <td>29.75</td>
    </tr>
  </tbody>
</table>
</div>




```python
taxi.shape
```




    (11225063, 19)




```python
type(taxi)
```




    pandas.core.frame.DataFrame




```python
taxi['points'] = pd.DataFrame(list(zip(list(taxi['pickup_longitude']), list(taxi['pickup_latitude'])))).T.apply(Point)
```


```python
taxi.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>VendorID</th>
      <th>tpep_pickup_datetime</th>
      <th>tpep_dropoff_datetime</th>
      <th>passenger_count</th>
      <th>trip_distance</th>
      <th>pickup_longitude</th>
      <th>pickup_latitude</th>
      <th>RatecodeID</th>
      <th>store_and_fwd_flag</th>
      <th>dropoff_longitude</th>
      <th>dropoff_latitude</th>
      <th>payment_type</th>
      <th>fare_amount</th>
      <th>extra</th>
      <th>mta_tax</th>
      <th>tip_amount</th>
      <th>tolls_amount</th>
      <th>improvement_surcharge</th>
      <th>total_amount</th>
      <th>points</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>2015-09-01 00:05:55</td>
      <td>2015-09-01 00:31:02</td>
      <td>1</td>
      <td>17.45</td>
      <td>-73.791351</td>
      <td>40.646690</td>
      <td>1</td>
      <td>N</td>
      <td>-73.857437</td>
      <td>40.848263</td>
      <td>1</td>
      <td>47.5</td>
      <td>0.5</td>
      <td>0.5</td>
      <td>5.00</td>
      <td>5.54</td>
      <td>0.3</td>
      <td>59.34</td>
      <td>POINT (-73.79135131835938 40.64669036865234)</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>2015-09-01 00:05:56</td>
      <td>2015-09-01 00:07:42</td>
      <td>1</td>
      <td>0.40</td>
      <td>-73.978935</td>
      <td>40.752853</td>
      <td>1</td>
      <td>N</td>
      <td>-73.986061</td>
      <td>40.755398</td>
      <td>2</td>
      <td>3.5</td>
      <td>0.5</td>
      <td>0.5</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.3</td>
      <td>4.80</td>
      <td>POINT (-73.97893524169923 40.75285339355469)</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>2015-09-01 00:05:57</td>
      <td>2015-09-01 00:16:48</td>
      <td>1</td>
      <td>1.50</td>
      <td>-73.990891</td>
      <td>40.723972</td>
      <td>1</td>
      <td>N</td>
      <td>-74.009560</td>
      <td>40.728916</td>
      <td>2</td>
      <td>9.0</td>
      <td>0.5</td>
      <td>0.5</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.3</td>
      <td>10.30</td>
      <td>POINT (-73.9908905029297 40.72397232055664)</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>2015-09-01 00:05:57</td>
      <td>2015-09-01 00:05:57</td>
      <td>1</td>
      <td>0.00</td>
      <td>-73.932655</td>
      <td>40.803768</td>
      <td>1</td>
      <td>N</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>2</td>
      <td>4.0</td>
      <td>0.5</td>
      <td>0.5</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.3</td>
      <td>5.30</td>
      <td>POINT (-73.93265533447266 40.80376815795898)</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>2015-09-01 00:05:57</td>
      <td>2015-09-01 00:30:32</td>
      <td>1</td>
      <td>7.50</td>
      <td>-73.987778</td>
      <td>40.738194</td>
      <td>1</td>
      <td>N</td>
      <td>-73.944756</td>
      <td>40.828167</td>
      <td>1</td>
      <td>23.5</td>
      <td>0.5</td>
      <td>0.5</td>
      <td>4.95</td>
      <td>0.00</td>
      <td>0.3</td>
      <td>29.75</td>
      <td>POINT (-73.98777770996094 40.73819351196289)</td>
    </tr>
  </tbody>
</table>
</div>




```python
taxi.tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>VendorID</th>
      <th>tpep_pickup_datetime</th>
      <th>tpep_dropoff_datetime</th>
      <th>passenger_count</th>
      <th>trip_distance</th>
      <th>pickup_longitude</th>
      <th>pickup_latitude</th>
      <th>RatecodeID</th>
      <th>store_and_fwd_flag</th>
      <th>dropoff_longitude</th>
      <th>dropoff_latitude</th>
      <th>payment_type</th>
      <th>fare_amount</th>
      <th>extra</th>
      <th>mta_tax</th>
      <th>tip_amount</th>
      <th>tolls_amount</th>
      <th>improvement_surcharge</th>
      <th>total_amount</th>
      <th>points</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>11225058</th>
      <td>1</td>
      <td>2015-09-21 15:35:03</td>
      <td>2015-09-21 15:42:09</td>
      <td>3</td>
      <td>0.9</td>
      <td>-73.998657</td>
      <td>40.725956</td>
      <td>1</td>
      <td>N</td>
      <td>-73.989021</td>
      <td>40.730812</td>
      <td>1</td>
      <td>6.5</td>
      <td>0.0</td>
      <td>0.5</td>
      <td>1.00</td>
      <td>0.0</td>
      <td>0.3</td>
      <td>8.30</td>
      <td>POINT (-73.9986572265625 40.72595596313477)</td>
    </tr>
    <tr>
      <th>11225059</th>
      <td>1</td>
      <td>2015-09-21 15:50:50</td>
      <td>2015-09-21 15:56:18</td>
      <td>3</td>
      <td>0.8</td>
      <td>-73.988388</td>
      <td>40.737949</td>
      <td>1</td>
      <td>N</td>
      <td>-73.975273</td>
      <td>40.733006</td>
      <td>2</td>
      <td>5.5</td>
      <td>0.0</td>
      <td>0.5</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.3</td>
      <td>6.30</td>
      <td>POINT (-73.98838806152342 40.73794937133789)</td>
    </tr>
    <tr>
      <th>11225060</th>
      <td>1</td>
      <td>2015-09-21 15:59:49</td>
      <td>2015-09-21 16:18:16</td>
      <td>1</td>
      <td>3.6</td>
      <td>-73.973442</td>
      <td>40.738232</td>
      <td>1</td>
      <td>N</td>
      <td>-73.975998</td>
      <td>40.776085</td>
      <td>1</td>
      <td>15.0</td>
      <td>0.0</td>
      <td>0.5</td>
      <td>3.16</td>
      <td>0.0</td>
      <td>0.3</td>
      <td>18.96</td>
      <td>POINT (-73.97344207763672 40.73823165893555)</td>
    </tr>
    <tr>
      <th>11225061</th>
      <td>1</td>
      <td>2015-09-21 16:22:44</td>
      <td>2015-09-21 16:24:38</td>
      <td>1</td>
      <td>0.5</td>
      <td>-73.982498</td>
      <td>40.782532</td>
      <td>1</td>
      <td>N</td>
      <td>-73.978195</td>
      <td>40.788685</td>
      <td>2</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>0.5</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.3</td>
      <td>4.80</td>
      <td>POINT (-73.98249816894531 40.78253173828125)</td>
    </tr>
    <tr>
      <th>11225062</th>
      <td>1</td>
      <td>2015-09-21 16:36:46</td>
      <td>2015-11-28 19:35:05</td>
      <td>1</td>
      <td>0.5</td>
      <td>-73.970222</td>
      <td>40.799171</td>
      <td>1</td>
      <td>N</td>
      <td>-73.969467</td>
      <td>40.794445</td>
      <td>3</td>
      <td>2.5</td>
      <td>0.0</td>
      <td>0.5</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.3</td>
      <td>3.30</td>
      <td>POINT (-73.97022247314453 40.79917144775391)</td>
    </tr>
  </tbody>
</table>
</div>




```python
taxi_gdf = gpd.GeoDataFrame(taxi, crs = {'init' :'epsg:4326'}, geometry = 'points')
```


```python
type(taxi_gdf), taxi_gdf.shape
```




    (geopandas.geodataframe.GeoDataFrame, (11225063, 20))




```python
taxi_join_df = gpd.sjoin(taxi_gdf, gdf, how = 'left', op = 'within')
# using a %%timeit command, this takes ~30mins
```


```python
taxi_join_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>VendorID</th>
      <th>tpep_pickup_datetime</th>
      <th>tpep_dropoff_datetime</th>
      <th>passenger_count</th>
      <th>trip_distance</th>
      <th>pickup_longitude</th>
      <th>pickup_latitude</th>
      <th>RatecodeID</th>
      <th>store_and_fwd_flag</th>
      <th>dropoff_longitude</th>
      <th>...</th>
      <th>points</th>
      <th>index_right</th>
      <th>boro_name</th>
      <th>boro_code</th>
      <th>shape_leng</th>
      <th>shape_area</th>
      <th>area_series</th>
      <th>area_series_ratio</th>
      <th>centroid</th>
      <th>centroid coords</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>2015-09-01 00:05:55</td>
      <td>2015-09-01 00:31:02</td>
      <td>1</td>
      <td>17.45</td>
      <td>-73.791351</td>
      <td>40.646690</td>
      <td>1</td>
      <td>N</td>
      <td>-73.857437</td>
      <td>...</td>
      <td>POINT (-73.79135131835938 40.64669036865234)</td>
      <td>4.0</td>
      <td>Queens</td>
      <td>4.0</td>
      <td>895228.960360</td>
      <td>3.044772e+09</td>
      <td>0.030143</td>
      <td>1.010102e+11</td>
      <td>POINT (-73.81849453624231 40.70760815187309)</td>
      <td>(-73.81849453624231, 40.70760815187309)</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>2015-09-01 00:05:56</td>
      <td>2015-09-01 00:07:42</td>
      <td>1</td>
      <td>0.40</td>
      <td>-73.978935</td>
      <td>40.752853</td>
      <td>1</td>
      <td>N</td>
      <td>-73.986061</td>
      <td>...</td>
      <td>POINT (-73.97893524169923 40.75285339355469)</td>
      <td>0.0</td>
      <td>Manhattan</td>
      <td>1.0</td>
      <td>361649.881587</td>
      <td>6.366006e+08</td>
      <td>0.006309</td>
      <td>1.009057e+11</td>
      <td>POINT (-73.96716968771112 40.77726401016272)</td>
      <td>(-73.96716968771112, 40.77726401016272)</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>2015-09-01 00:05:57</td>
      <td>2015-09-01 00:16:48</td>
      <td>1</td>
      <td>1.50</td>
      <td>-73.990891</td>
      <td>40.723972</td>
      <td>1</td>
      <td>N</td>
      <td>-74.009560</td>
      <td>...</td>
      <td>POINT (-73.9908905029297 40.72397232055664)</td>
      <td>0.0</td>
      <td>Manhattan</td>
      <td>1.0</td>
      <td>361649.881587</td>
      <td>6.366006e+08</td>
      <td>0.006309</td>
      <td>1.009057e+11</td>
      <td>POINT (-73.96716968771112 40.77726401016272)</td>
      <td>(-73.96716968771112, 40.77726401016272)</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>2015-09-01 00:05:57</td>
      <td>2015-09-01 00:05:57</td>
      <td>1</td>
      <td>0.00</td>
      <td>-73.932655</td>
      <td>40.803768</td>
      <td>1</td>
      <td>N</td>
      <td>0.000000</td>
      <td>...</td>
      <td>POINT (-73.93265533447266 40.80376815795898)</td>
      <td>0.0</td>
      <td>Manhattan</td>
      <td>1.0</td>
      <td>361649.881587</td>
      <td>6.366006e+08</td>
      <td>0.006309</td>
      <td>1.009057e+11</td>
      <td>POINT (-73.96716968771112 40.77726401016272)</td>
      <td>(-73.96716968771112, 40.77726401016272)</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>2015-09-01 00:05:57</td>
      <td>2015-09-01 00:30:32</td>
      <td>1</td>
      <td>7.50</td>
      <td>-73.987778</td>
      <td>40.738194</td>
      <td>1</td>
      <td>N</td>
      <td>-73.944756</td>
      <td>...</td>
      <td>POINT (-73.98777770996094 40.73819351196289)</td>
      <td>0.0</td>
      <td>Manhattan</td>
      <td>1.0</td>
      <td>361649.881587</td>
      <td>6.366006e+08</td>
      <td>0.006309</td>
      <td>1.009057e+11</td>
      <td>POINT (-73.96716968771112 40.77726401016272)</td>
      <td>(-73.96716968771112, 40.77726401016272)</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 29 columns</p>
</div>




```python
pickup_frequency = taxi_join_df['boro_name'].value_counts()
```


```python
gdf['pickup_frequency'] = pickup_frequency.values
```


```python
x_index = 0
y_index = 1
offset = 0.001

fig, ax = plt.subplots(figsize = (8,5))
gdf.plot(ax = ax, column = 'pickup_frequency', cmap = 'Blues', scheme = 'quantiles') # legend = True
gdf["geometry"].centroid.plot(ax = ax, c = 'b')
for i, label in enumerate(gdf["boro_name"]):
    ax.text(gdf["centroid coords"][i][x_index]+ offset, gdf["centroid coords"][i][y_index], 
            f"{label}\n Pickups = {gdf['pickup_frequency'][i]:,.0f}", 
            size=8, rotation=45.,
             ha="left", va="bottom",
             )
```


![png](/images/starter-code-2.0_files/starter-code-2.0_90_0.png)


#### Suppose we want to forecast the number of pick-ups by borough. Would this process be described as areal, geostatistical, or point pattern?


```python
# Areal process, because we have non-overlapping regions
```

#### Bonus: Build a widget that will put dots on the map for the location of each pick-up by date.
Using the exact latitude and longitude will cause multiple dots to overlap; people often use a [random jitter](https://www.dataplusscience.com/TableauJitter.html) to help with this. While not required, consider random jitter as an extra bonus!

#### In order to predict the precise location of pick-ups, would this process be described as areal, geostatistical, or point pattern?


```python
# Point pattern process, because these are a collection of random points
```


```python
x_index = 0
y_index = 1
offset = 0.005

fig, ax = plt.subplots(figsize = (8,5))
plt.title('\n Simulated taxi pickups in NYC v. actual pickup frequency \n ')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
gdf.plot(ax = ax, column = 'pickup_frequency', cmap = 'cool', scheme = 'quantiles') # legend = True
random_points_gdf['points'].plot(ax = ax, markersize = 0.3, c = 'b', label = 'Simulated points')
gdf["geometry"].centroid.plot(ax = ax, c = 'b')
for i, label in enumerate(gdf["boro_name"]):
    ax.text(gdf["centroid coords"][i][x_index]+ offset, gdf["centroid coords"][i][y_index], 
            f"{label}\n Pickups = {gdf['pickup_frequency'][i]:,.0f}", 

            size=8, rotation=45.,
            ha="left", va="bottom",
            bbox=dict(facecolor = 'white', edgecolor='none')
             )
plt.legend(facecolor = 'white', framealpha = 1);
```


![png](/images/starter-code-2.0_files/starter-code-2.0_96_0.png)

