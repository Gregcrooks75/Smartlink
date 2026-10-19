
import geopandas as gpd
from shapely.geometry import Polygon, Point, LineString, MultiLineString, MultiPolygon


def getPolyCoords(row, geom, coord_type):
    """
    Returns the coordinates ('x' or 'y') of edges/vertices of a Polygon or others.

    Args:
        - row: the row object from a geodataframe; i.e. df.loc[1]
        - geom: the name of "geometry" column, usually "geometry"
        - coord_type: 'x' or 'y'

    Returns:
        A list of coordinates (x or y)
    """

    geometry = row[geom]

    if geometry.type == 'Polygon':
        if coord_type == 'x':
            return list(geometry.exterior.coords.xy[0])
        elif coord_type == 'y':
            return list(geometry.exterior.coords.xy[1])
    elif geometry.type in ['Point', 'LineString']:
        if coord_type == 'x':
            return list(geometry.xy[0])
        elif coord_type == 'y':
            return list(geometry.xy[1])
    elif geometry.type == 'MultiLineString':
        all_xy = []
        for ea in geometry.geoms:
            if coord_type == 'x':
                all_xy.append(list(ea.xy[0]))
            elif coord_type == 'y':
                all_xy.append(list(ea.xy[1]))
        return all_xy
    elif geometry.type == 'MultiPolygon':
        all_xy = []
        for ea in geometry.geoms:
            if coord_type == 'x':
                all_xy.append(list(ea.exterior.coords.xy[0]))
            elif coord_type == 'y':
                all_xy.append(list(ea.exterior.coords.xy[1]))
        return all_xy
    else:
        return []
    
if __name__ == '__main__':
    df = getPolyCoords(row, geom, coord_type)