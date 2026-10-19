import folium
from shapely.geometry import Point, LineString, Polygon, MultiPolygon
import geopandas as gpd
def get_or_create_feature_group(name, mapObj):
    fg = folium.FeatureGroup(name=name)

    return fg.add_to(mapObj)


    return fg.add_to(mapObj)
def add_simple_legend(map_obj, title, items):
    """
    Adds a simple legend to a folium map.

    Parameters:
    - map_obj: folium.Map object
    - title: str, title of the legend
    - items: list of dicts like:
        {'label': 'Meters', 'kind': 'point', 'color': 'blue'}
        {'label': 'Good — Meter→Transformer', 'kind': 'line', 'color': 'blue'}
        {'label': 'Meters Area', 'kind': 'polygon', 'color': '#777'}
    """
    rows = []
    for it in items:
        label = it.get('label', '')
        color = it.get('color', '#000')
        kind = it.get('kind', 'point')

        if kind == 'point':
            swatch = f'<i style="background:{color}; border-radius:50%; width:10px; height:10px; display:inline-block; margin-right:6px;"></i>'
        elif kind == 'line':
            swatch = f'<i style="background:{color}; width:20px; height:2px; display:inline-block; margin-right:6px;"></i>'
        elif kind == 'polygon':
             swatch = f'<i style="background:rgba(0, 0, 255, 0.1); width:10px; height:10px; display:inline-block; margin-right:6px; border:1px solid blue;"></i>'
        else:
            swatch = ''

        rows.append(f'<div style="margin-bottom:4px;">{swatch}{label}</div>')

    html = f"""
    <div style="
        position: fixed;
        bottom: 30px;
        left: 30px;
        z-index: 9999;
        background-color: white;
        padding: 10px;
        border: 2px solid grey;
        border-radius: 5px;
        box-shadow: 2px 2px 6px rgba(0,0,0,0.3);
        font-size: 12px;
    ">
        <strong>{title}</strong>
        <div style="margin-top: 5px;">
            {''.join(rows)}
        </div>
    </div>
    """

    macro = MacroElement()
    macro._template = Template(f"""
        {{% macro html(this, kwargs) %}}
            {html}
        {{% endmacro %}}
    """)
    map_obj.get_root().add_child(macro)

def create_points(
    lat_col,
    long_col,
    icon,
    icon_color,
    df,
    metadata_fields,
    layer_name,
    mapObj,
    alias_map=None
):
    """
    Adds markers with popups to a Folium FeatureGroup.

    Parameters:
    - lat_col, long_col: Coordinate columns (latitude and longitude).
    - icon, icon_color: Folium icon settings.
    - df: GeoDataFrame with spatial data.
    - metadata_fields: List of fields to display in the popup.
    - layer_name: Name of the FeatureGroup to attach to.
    - mapObj: The base Folium map object.
    - alias_map: (Optional) Dictionary mapping field names to user-friendly labels.
    """
    fg = get_or_create_feature_group(layer_name, mapObj)

    if df.empty:
        print(f"[SKIPPED] No data for layer '{layer_name}'")
        return

    for _, row in df.iterrows():
        try:
            lat = float(row[lat_col])
            lon = float(row[long_col])
        except Exception as e:
            print(f"[SKIP POINT] Invalid coords: {e}")
            continue

        popup_text = ""

        for field in metadata_fields:
            label = alias_map.get(field, field) if alias_map else field
            value = row.get(field, "N/A")

        # 💡 Check if field looks like a link field
            if field.lower() in ["googlemaps", "streetview", "link", "url"]:
                if isinstance(value, str) and value not in ["N/A", ""]:
                    popup_text += f'<b>{label}:</b> <a href="{value}" target="_blank">Open</a><br>'
                else:
                    popup_text += f"<b>{label}:</b> N/A<br>"
            else:
                popup_text += f"<b>{label}:</b> {value}<br>"

        popup = folium.Popup(popup_text, max_width=300, min_width=150)

        folium.Marker(
            location=[lat, lon],
            icon=folium.Icon(icon=icon, prefix="fa", color=icon_color),
            popup=popup
        ).add_to(fg)

    print(f"[ADDED] {len(df)} points to '{layer_name}'")



def create_line(
    df,
    geometry,
    style_function,
    layer_name,
    mapObj
):
    """
    Adds a GeoJson line from a geometry column to a named FeatureGroup on the map.

    Parameters:
    - df: GeoDataFrame containing the geometry.
    - geometry: Column name containing LineString geometries.
    - style_function: Function defining the visual style for the GeoJson lines.
    - layer_name: Name of the FeatureGroup layer.
    - mapObj: The base Folium map object.
    """
    fg = get_or_create_feature_group(layer_name, mapObj)

    if 'LineString' in df[geometry].apply(lambda geom: geom.geom_type).unique():
        geojson = gpd.GeoDataFrame(geometry=df[geometry]).to_json()
        folium.GeoJson(
            geojson,
            style_function=style_function
        ).add_to(fg)

def make_style_function(
    color='black',
    weight=2,
    fill=False,
    fill_opacity=0.2,
    line_opacity=1.0
):
    """
    Returns a style function for folium GeoJson.

    Parameters:
    - color: Line and fill color (default: 'black').
    - weight: Stroke width in pixels (default: 2).
    - fill: Boolean indicating whether to fill the shape (default: False).
    - fill_opacity: Opacity of the fill area (default: 0.2).
    - line_opacity: Opacity of the line stroke (default: 1.0).
    """
    
    def style_fn(feature):
        return {
            "color": color,
            "weight": weight,
            "opacity": line_opacity,
            "fillColor": color if fill else "none",
            "fillOpacity": fill_opacity if fill else 0
        }

    return style_fn

from branca.element import MacroElement
from jinja2 import Template

def classify_geometry_type(gdf):
    geom_types = gdf.geometry.geom_type.unique()
    if "Point" in geom_types or "MultiPoint" in geom_types:
        return "Point"
    elif "LineString" in geom_types or "MultiLineString" in geom_types:
        return "Line"
    elif "Polygon" in geom_types or "MultiPolygon" in geom_types:
        return "Polygon"
    else:
        return "Unknown"

def add_auto_legend_from_layers(mapObj, layer_data, title="Map Legend"):
    html = f"""
    {{% macro html(this, kwargs) %}}

    <div style='position: fixed;
                bottom: 50px; left: 50px; width: 310px; height: auto;
                z-index: 9999; font-size: 14px;
                background-color: white; padding: 12px;
                border: 2px solid gray; border-radius: 8px;
                box-shadow: 2px 2px 5px rgba(0,0,0,0.3);'>

        <b>{title}</b><br><br>
    """

    for name, info in layer_data.items():
        geom_type = info.get("geometry")
        if isinstance(info.get("data"), gpd.GeoDataFrame):
            geom_type = classify_geometry_type(info["data"])

        color = info.get("color", "#000000")
        icon = info.get("icon", "fa-circle")

        if geom_type == "Point":
            html += f"""
            <div style="display: flex; align-items: center; margin-bottom: 6px;">
                <div style="background-color: {color}; width: 26px; height: 26px;
                            border-radius: 50% 50% 50% 0; transform: rotate(-45deg);
                            text-align: center; color: white; line-height: 26px; font-size: 12px;">
                    <i class="fa {icon}" style="transform: rotate(45deg);"></i>
                </div>
                <span style="margin-left: 10px;">{name}</span>
            </div>
            """
        elif geom_type == "Line":
            html += f"""
            <div style="display: flex; align-items: center; margin-bottom: 6px;">
                <svg width="20" height="10">
                    <line x1="0" y1="5" x2="20" y2="5" style="stroke:{color};stroke-width:2"/>
                </svg>
                <span style="margin-left: 10px;">{name}</span>
            </div>
            """
        elif geom_type == "Polygon":
            html += f"""
            <div style="display: flex; align-items: center; margin-bottom: 6px;">
                <svg width="20" height="20">
                    <circle cx="10" cy="10" r="8"
                            style="fill:{color}; stroke:black; stroke-width:1; fill-opacity:0.3"/>
                </svg>
                <span style="margin-left: 10px;">{name}</span>
            </div>
            """

    html += "</div>{% endmacro %}"

    legend = MacroElement()
    legend._template = Template(html)
    mapObj.get_root().add_child(legend)