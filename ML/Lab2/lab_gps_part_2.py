#!/usr/bin/env python
# coding: utf-8

# # Gaussian process for geographic data (GP lab part 2)
# 
# **Machine Learning, University of Zaragoza, Ruben Martinez-Cantin**
# 
# In this lab, you will implement a Gaussian process for air quality data, obtained from the City Hall in their [open data site](https://www.zaragoza.es/sede/portal/datos-abiertos/servicio/catalogo/131).
# 
#   *adapted from a lab by Luis Montesano*

# In[ ]:


#@title install libraries
# !pip install GPy


# In[7]:


#@title import libraries

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from geopy import Nominatim, Photon
from PIL import Image
from pyproj import Transformer
#from sklearn.gaussian_process import GaussianProcessRegressor
#from sklearn.gaussian_process.kernels import RBF, WhiteKernel
import GPy

# descomenta %matplotlib qt si prefieres plots interactivos
get_ipython().run_line_magic('matplotlib', 'inline')
# %matplotlib qt


# ## 0. Setup y Auxiliary Funtions

# ### 0.1 Coordinate transformation
# 
# The geographical coordinates we will use are given in degrees (*latitude* and *longitude*). Therefore, we need a way to transform them (or *project* them) to metric units (e.g. meters), as these are more suitable in the calculation of distances needed in **Gaussian processes**.
# 
# For these transformations, we will use the **PyProj** library. [ [code](https://github.com/pyproj4/pyproj) | [docs](https://pyproj4.github.io/pyproj/stable/) ].
# 
# 
# &#9432; **EXTRA INFO**
# 
# These transformations are performed using the following **coordinate reference systems (CRS)**:
# * EPSG:4326 (WGS84) as being the appropriate CRS for latitude and longitude coordinates (represented in degrees) [<a href="https://epsg.io/?q=4326">link</a>].
# * EPSG:32630 (UTM, zona 30) as this CRS is used to express the coordinates in meters, projecting them to a local 2D plane. Zone 30 is the one corresponding to Zaragoza [<a href="https://www.ign.es/web/coordenadas-de-stations-ergnss">link</a>].
# 

# In[3]:


transformer_longlat_xy = Transformer.from_crs(4326, 32630, always_xy=True)


def from_longlat_to_xy(long, lat):
    """Transform longitude and latitude coordinates to x, y coordinates in
    CRS 32630 (UTM, zone 30)

    Args:
        long: (n,) array or float with longitude (degrees)
        lat: (n,) array or float with latitude (degrees)

    Returns:
        x: (n,) array or float with x coordinates in CRS 32630
        y: (n,) array or float with y coordinates in CRS 32630
    """
    return transformer_longlat_xy.transform(long, lat)


def from_xy_to_longlat(x, y):
    """Transform x, y coordinates in CRS 32630 (UTM, zone 30) to longitude and
    latitude coordinates

    Args:
        x: (n,) array or float with x coordinates in CRS 32630
        y: (n,) array or float with y coordinates in CRS 32630

    Returns:
        long: (n,) array or float with longitude (degrees)
        lat: (n,) array or float with latitude (degrees)
    """
    return transformer_longlat_xy.transform(x, y, direction="inverse")


# ### 0.2 Generation and data loading
# 

# First, we need to obtain the **geographic coordinates** (latitude and longitude) of the air quality measurement **stations** in Zaragoza.
# 
# Knowing the station location (street or address) for each station, we use `Photon` *geodecoder* to obtain an automated (albeit approximate) location of the stations using `OpenStreetMap` data. For that, we use the `Geopy` library [[*GH*](https://github.com/geopy/geopy) | [*Docs*](https://geopy.readthedocs.io/en/stable/)].
# 
# The address of each station can be found in the city hall [website](https://www.zaragoza.es/sede/portal/medioambiente/calidad-aire/). We store the data in a `Pandas` dataframe to simplify access and to show the results in a table form.

# In[8]:


stations = {
    "Actur": "Calle Cineasta Carlos Saura, 50018",
    "El Picarral": "Av. San Juan de la Peña / C.S. Picarral, 50015",
    "Roger de Flor": "Calle Roger de Flor, 50017",
    "Jaime Ferran": "Calle Jaime Ferrán, 50014",
    "Renovales": "Paseo de Mariano Renovales, 50006",
    "Las Fuentes": "Calle María de Aragón, 50013",
    "Centro": "Calle José Luis Albareda, 50004",
    "Avda. Soria": "Avenida Ciudad de Soria, 50010, Zaragoza, España",
}

latitudes, longitudes = [], []
geolocator = Photon(user_agent="myGeocoder",timeout=100)
for name, address in stations.items():
    location = geolocator.geocode(address)
    latitudes.append(location.latitude)  # type: ignore
    longitudes.append(location.longitude)  # type: ignore

data = {
    "station": list(stations.keys()),
    "latitude": latitudes,
    "longitude": longitudes,
}
df_stations = pd.DataFrame(data)

df_stations


# Now, let's add the **metric coordinates**  $x, y$ (meters) corresponding to the *CRS UTM (zone 30)* of each station:

# In[5]:


df_stations[["x", "y"]] = df_stations.apply(
    lambda row: from_longlat_to_xy(row.longitude, row.latitude),
    axis=1,
    result_type="expand",
)
df_stations


# Last, we load the file `calidad_aire_2023.csv` corresponding to the air quality measured as the *concentration* [$\mu g/m^3$] of several chemical contaminants, measured in each station through 2023.
# 
# If you are working locally in your system or you already have downloaded the file, you can skip this step.

# In[ ]:


#@title download data
#!wget https://raw.githubusercontent.com/rmcantin/gp_lab/main/datasets/calidad_aire_2023.csv


# In[7]:


#@title load downloaded data
path_calidad = "calidad_aire_2023.csv"
df_air_q = pd.read_csv(path_calidad, sep=",")
df_air_q


# ### 0.3 Interactive visualization

# In the next code, we develop the class `MapPlotter`, based in [**Plotly**](https://plotly.com/graphing-libraries/) (using the [map visualization API](https://plotly.com/python/mapbox-layers/)) which defines auxiliar methods to visualize the lab session results.

# In[15]:


class MapPlotter:
    """Interactive map visualization based on Plotly"""

    CENTER = {"lat": 41.65573, "lon": -0.88616}

    def __init__(self, zoom=12, center=None, showlegend=False, **kwargs):
        self.fig = go.Figure()
        self.fig.update_layout(
            mapbox_style="open-street-map",  # or "carto-positron",
            mapbox_zoom=zoom,
            mapbox_center=center or self.CENTER,
            margin={"r": 0, "t": 10, "l": 0, "b": 10},
            showlegend=showlegend,
            # autosize=True,
            **kwargs,
        )

    def add_longlat_points(
        self,
        lon,
        lat,
        s=15,
        color=None,
        text=None,
        edgecolor=None,
        name=None,
        adjust_center=True,
        **kwargs,
    ):
        if edgecolor is not None:
            self.fig.add_trace(
                go.Scattermapbox(
                    lat=lat,
                    lon=lon,
                    mode="markers",
                    marker_size=1.1 * s,
                    marker_color=edgecolor,
                    showlegend=False,
                    **kwargs,
                )
            )

        self.fig.add_trace(
            go.Scattermapbox(
                lat=lat,
                lon=lon,
                mode="markers",
                marker_size=s,
                marker_color=color,
                text=text,
                name=name,
                **kwargs,
            )
        )

        if adjust_center:
            self.adjust_center(lon.mean(), lat.mean())

    def add_density_heatmap(self, lon, lat, z, radius=10, adjust_center=True, **kwargs):
        self.fig.add_trace(
            go.Densitymapbox(
                lat=lat.ravel(),
                lon=lon.ravel(),
                z=z.ravel(),
                radius=radius,
                colorbar_title="μg/m³",
                **kwargs,
            )
        )

        if adjust_center:
            self.adjust_center(lon.mean(), lat.mean())

    def add_raster_heatmap(
        self,
        xgrid,
        ygrid,
        heatmap,
        cmap="Spectral_r",
        adjust_center=True,
        add_cbar=True,
        **kwargs,
    ):
        # rasterize the heatmap
        heatmap_ = (heatmap - heatmap.min()) / np.ptp(heatmap)
        heatmap_ = plt.get_cmap(cmap)(heatmap_)[..., :3]
        raster_hm = Image.fromarray((heatmap_ * 255).clip(0, 255).astype(np.uint8))

        # heatmap bounding box
        bbox = np.array(
            [
                [xgrid[0, 0], ygrid[0, 0]],  # top-left
                [xgrid[0, -1], ygrid[0, -1]],  # top-right
                [xgrid[-1, -1], ygrid[-1, -1]],  # bottom-right
                [xgrid[-1, 0], ygrid[-1, 0]],  # bottom-left
            ]
        )
        lon, lat = from_xy_to_longlat(bbox[:, 0], bbox[:, 1])
        coords = np.stack((lon, lat), axis=1)

        layers = [*self.fig.layout.mapbox.layers]  # type: ignore
        layers.append(
            {
                "sourcetype": "image",
                "source": raster_hm,
                "coordinates": coords,
                "below": "traces",
                **kwargs,
            }
        )
        self.fig.layout.mapbox.layers = layers  # type: ignore

        if adjust_center:
            self.adjust_center(lon.mean(), lat.mean())

        if add_cbar:
            # add colorbar via invisible scatter trace
            self.fig.add_trace(
                go.Scattermapbox(
                    lat=[None],
                    lon=[None],
                    mode="markers",
                    marker=dict(
                        colorscale=cmap,
                        color=heatmap,
                        cmin=heatmap.min(),
                        cmax=heatmap.max(),
                        showscale=True,
                        colorbar=dict(
                            title="μg/m³",
                            # titleside="right",
                            ticks="outside",
                        ),
                    ),
                )
            )

    def adjust_center(self, lon, lat):
        self.fig.update_layout(mapbox_center={"lat": lat, "lon": lon})

    def show(self):
        self.fig.show()


# ## 1. Gaussian process for geographical data

# In this part of the lab, you must implement the **Gaussian process model** to interpolate the air quaility in Zaragoza and visualize the results obtained.
# 
# For this, you need to complete the following functions:
# 
# &#9432; **WARNING**
# 
# Some stations do not have concentration data for all the contaminants, because they lack the corresponding sensor. We recommend you to use the following contaminants which are available in all the stations:
# 
# * `nox`
# * `no2`
# * `no`
# * `o3`

# In[16]:


#@title Utils to process the data
def create_regular_grid(X, npts=100):
    """Coordinate mesh grid of where to predict the air quaility using GPs

    Args:
        X: (nobs, 2) array of the xy coordinates of the observations
        npts: number of points in width and height in the grid.

    Returns:
        xgrid: (npts, npts) array with x coordinates of the grid
        ygrid: (npts, npts) array with y coordinates of the grid
    """
    # grid limits
    x_min, x_max = X[..., 0].min(), X[..., 0].max()
    y_min, y_max = X[..., 1].min(), X[..., 1].max()

    # we add 10% padding/margin
    padding_x = (x_max - x_min) * 0.1
    padding_y = (y_max - y_min) * 0.1

    x_min -= padding_x
    x_max += padding_x
    y_min -= padding_y
    y_max += padding_y

    # regular grid
    x, y = np.linspace(x_min, x_max, npts), np.linspace(y_min, y_max, npts)
    xgrid, ygrid = np.meshgrid(x, y)
    return xgrid, ygrid

def load_air_quality_data(df_air_q, contaminant="o3", log_data=False):
    """Load air quality data for GP estimation in Zaragoza

    Args:
        df_air_q: Air quality DataFrame
        contaminant: str, contaminant name to estimate
        log_data: if True, print the observations of that contaminant in each
            station

    Returns:
        X: (n_stations, 2) array with the location of the stations
        measured_concentration: (n_stations, ) array with the observations for
            contaminant at the stations (average of 2023).
    """
    # first, we check that the contamint is in the DataFrame
    assert (
        contaminant in df_air_q["contaminante"].unique()
    ), "Contaminant not found"

    # we filter all data regarding the contamint
    data_contaminante = df_air_q[df_air_q["contaminante"] == contaminant]

    # As observation, we use the mean value along 2023 of each contamint in
    # each station
    observations = data_contaminante.groupby("title")["value"].mean()

    # We create an auxiliary dataframe where we add the observation to each station
    df_est = df_stations.copy()
    df_est["concen."] = df_est["station"].map(observations)
    assert (
        df_est["concen."].notnull().all()
    ), "Missing observations for some stations"

    if log_data:
        print(df_est)

    # Input data for the Gaussian process
    X = df_est[["x", "y"]].values    # location of the stations
    measured_concentrations = df_est["concen."].values

    return X, measured_concentrations


# In[18]:


#@title Function to predict concentration based on a Gaussian process
#@markdown Complete the TODO sections


def air_quality_prediction(X, measured_concentrations, gp_kernel, npts_grid=100):
    """GP-based air quality estimation in Zaragoza

    Args:
        X: (n_stations, 2) array with the location of the stations
        measured_concentration: (n_stations, ) array with the observations for
            contaminant at the stations (average of 2023).
        gp_kernel: Kernel for the GP to train
        npts_grid: int, number of points in width and height in the grid where
            we are going to predict the air quality

    Returns:
        (xgrid, ygrid): tupla with all x,y coordinates of the grid in
            shape (npts, npts)
        pred_concentration: (npts, npts) array with the predicted mean value of
            the contamint at each point of the grid.
        sigma_concentration: (npts, npts) array with the predicted standard deviation
            of the contamint at each point of the grid.
    """

    # Gaussian process training using the observations
    # TODO: Add the training code here using GPy and gp_kernel
    #
    # IMPORTANT:
    #    Sometimes Python vectors are represented with an empty dimension (n, ).
    #    Many libraries consider verctors as a special case of a 1D matrix.
    #    Thus, we need to make the vector (n, 1). For that, you can use either:
    #     - numpy.atleast_2d
    #     - numpy.newaxis
    #    Check the documentation for each method. Note that (1, n) is a different
    #    matrix, but you can then transpose the result.
    # Reshape the measurements to a column vector
    X = np.atleast_2d(X)
    measured_concentrations = np.atleast_2d(measured_concentrations).T  # Make it (n_stations, 1)

    # Create and train the GP model using GPy
    model = GPy.models.GPRegression(X, measured_concentrations, gp_kernel)
    model.optimize(messages=True)
    display(model)


    # Create a regular mesh to predict the air quality in all the city.
    xgrid, ygrid = create_regular_grid(X, npts=npts_grid)

    # Gaussian process prediction at the grid points
    # TODO: Using the trained model, predict the concentration mean and standar
    # deviation at the grid locations
    #
    # IMPORTANT:
    #     You might need to reshape the values of the grid. GPy expects a single
    #     matrix with size (n_predictions, input_dim). To reshape the grids you
    #     can use the following methods:
    #     -numpy.ravel
    #     -numpy.vstack
    #     and you might also need to transpose the results.
    # Reshape the grid for predictions
    Xgrid = np.vstack([xgrid.ravel(), ygrid.ravel()]).T  # Reshape grid to (n_predictions, 2)



    # Reshape the GP predictions to adjust it to the grid: shape (npts, npts)
    # This will be required for plotting. Right now we are returning random values
    # that you can use for checking the plots.
    #
    # TODO: replace the random values with the GP prediction.
    #
    # IMPORTANT:
    #      For reshape, you can direcly use numpy.reshape. Also, check that you
    #      return the standard deviation and not the variance.

    # Comment these two lines and make your own predictions
    # pred_concentration = np.random.rand(npts_grid, npts_grid)
    # std_concentration = np.random.rand(npts_grid, npts_grid)

    # Predict the mean and standard deviation at the grid points
    pred_mean, pred_var = model.predict(Xgrid)

    # Reshape the predicted mean and standard deviation to match grid shape
    pred_concentration = pred_mean.reshape((npts_grid, npts_grid))
    std_concentration = np.sqrt(pred_var).reshape((npts_grid, npts_grid))  # Convert variance to std deviation



    return (xgrid, ygrid), pred_concentration, std_concentration


# ## Process data and Kernel definition
# 
# Using GPy, you need to define the kernel. Try different kernels and combinations of them.
# 
# IMPORTANT: GPy uses gradient descent to learn the hyperparameters. This might fail if the hyperparameters are quite far from the optimal values as they get stuck in local minima. You can solve this in two ways:
# - Either give an good initial value for the hyperparameters (order of magnitude). For example: as a rule of thumb, the lenghtscale should be in the same scale as the GP inputs and the variance should be in the same scale as the GP outputs.
# - Alternatively, if we assume that the original data follows a Gaussian distribution $X \sim N(\mu_X, \sigma_X)$, then you can normalize your data such that the mean value is 0 and the variance is 1, that is, $newX \sim N(0,1)$. Then:
# $$ newX = \frac{X - \mu_X}{\sigma_X}$$
# If you train your GP with the normalize data, then, you need to remember to apply the same factor to the prediction location
# 
# $$ new\_prediction = \frac{original\_prediction - \mu_X}{\sigma_X}$$
# 
# **Data normalization** is always a good practice if done properly and improves the numerical performance in **all kind of problems**.

# In[37]:


kernel = GPy.kern.RBF(input_dim=2, lengthscale=1000)

#Loading the data for the selected contaminant.
#TODO: Test different contaminants
X, y = load_air_quality_data(df_air_q, contaminant="no2")

#TODO: Normalize your data if needed

# Compute the predictions of the Gaussian process
grid, concentration_prediction, concentration_std = air_quality_prediction(X, y, kernel)

#TODO: Make sure the grid is NOT affected by the normalization.
#TODO: You can call create_regular_grid again if needed


# ## Visualization
# 
# Let's plot the prediction and standard deviation in two separate maps.
# 

# In[32]:


fig = MapPlotter()

# Adding the stations as points in the maps:
fig.add_longlat_points(
    df_stations["longitude"],
    df_stations["latitude"],
    text=df_stations.station,
    color="white",
    edgecolor="black",
)

# Adding the predicted concentration as a heatmap in the grid
fig.add_raster_heatmap(grid[0], grid[1], concentration_prediction, opacity=0.6)

# Show map
fig.show()


# In[33]:


fig2 = MapPlotter()

# Adding the stations as points in the maps:
fig2.add_longlat_points(
    df_stations["longitude"],
    df_stations["latitude"],
    text=df_stations.station,
    color="white",
    edgecolor="black",
)

# Adding the predicted concentration as a heatmap in the grid
fig2.add_raster_heatmap(grid[0], grid[1], concentration_std, opacity=0.6)

# Show map
fig2.show()


# # Your tasks
# 
# - Use different kernels to predict the concentration for each of the contaminants. Justify your selection.
# 
# - Explain how did you solve the problem with the hyperparameter learning.
# 
# - If you can invest in a new station. Where would you place it?
# zona roja, menos datos
