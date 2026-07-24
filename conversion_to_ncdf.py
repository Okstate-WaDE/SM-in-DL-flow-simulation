
import pandas as pd
import xarray as xr


df = pd.read_csv("/Users/gautam/Research/codesforresearch/percentsaturation/Voronoi_autoregressive/time_series/area_weighted_with_Q.csv")

xr_data = df.set_index(['date']).to_xarray()

# Save the xarray Dataset to a NetCDF file 
xr_data.to_netcdf("/Users/gautam/Research/codesforresearch/percentsaturation/Voronoi_autoregressive/time_series/Washita.nc")

