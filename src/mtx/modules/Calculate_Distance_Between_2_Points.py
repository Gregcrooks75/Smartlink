from pyproj import Geod
from shapely.geometry import Point, LineString, Polygon
import pandas as pd
import geopandas as gpd

def invert_coords(line):
    if line and not line.is_empty:
        return LineString([(lat, lon) for lon, lat in line.coords])
    else:
        return line

# Create LineString from comma-separated lat/lon strings
def create_geometry(lat_str, lon_str):
    if pd.isna(lat_str) or pd.isna(lon_str):
        return None

    latitudes = list(map(float, lat_str.split(',')))
    longitudes = list(map(float, lon_str.split(',')))

    if len(latitudes) >= 2:
        return LineString(zip(latitudes, longitudes))  # Correct order: (lon, lat)
    else:
        return Point(latitudes[0], longitudes[0])

def calculate_length(line):
    geod = Geod(ellps='WGS84')
    if line.geometry.type == 'MultiLineString':
        return sum(geod.geometry_length(geom) for geom in line.geomtry.geoms)
    else:
        return geod.geometry_length(line.geometry)


def find_line_and_distance(lon1, lon2, lat1, lat2):
    """
    Correct version:
    - Takes lon1, lon2, lat1, lat2 (in that order)
    - Builds LineString using (lon, lat)
    - Returns a DataFrame with 'Line' and 'Distance' in feet
    """
    geod = Geod(ellps='WGS84')
    results = {'Line': [], 'Distance': []}

    for lo1, lo2, la1, la2 in zip(lon1, lon2, lat1, lat2):
        if pd.notna(lo1) and pd.notna(lo2) and pd.notna(la1) and pd.notna(la2):
            # Create LineString in (lon, lat) format
            line = LineString([(lo1, la1), (lo2, la2)])
            distance = geod.geometry_length(line) * 3.28084  # Convert meters to feet
            results['Line'].append(line)
            results['Distance'].append(distance)
        else:
            results['Line'].append(None)
            results['Distance'].append(None)

    return pd.DataFrame(results)


def add_distance(df, lat1, lat2, lon1, lon2, name_prefix):
    df[[f'Line_{name_prefix}', f'Distance_{name_prefix}']] = find_line_and_distance(
        df[lon1],
        df[lon2],
        df[lat1],
        df[lat2]
    )
    return df

def create_point(df, lat_col, lon_col, new_col):
    df[new_col] = df.apply(
        lambda row: Point(row[lon_col], row[lat_col]),
        axis=1
    )
    return df

def Discrepancy_check(row, distance_col):
    distance = row[distance_col]

    if pd.isna(distance):
        return 'Missing Distance'
    
    if distance >= 550:
        return '550 Feet or above'
    else:
        return 'Under 150 Feet'

def create_buffer(df, x_col, y_col, epsg_in=4326, epsg_out=32619, radius_ft=550):
    df = df.copy()

    # Create geometry column using Point objects
    df['geometry'] = df.apply(
        lambda row: Point(row[y_col], row[x_col]),
        axis=1
    )

    # Convert to GeoDataFrame and reproject to target CRS
    gdf = gpd.GeoDataFrame(df, geometry='geometry', crs=f"EPSG:{epsg_in}")
    gdf = gdf.to_crs(epsg=epsg_out)

    # Create buffer (radius converted from feet to meters)
    gdf['geometry'] = gdf.buffer(radius_ft * 0.3048)

    # Reproject back and return geometry as GeoJSON
    return gdf.to_crs(epsg=epsg_in)['geometry'].to_json()


