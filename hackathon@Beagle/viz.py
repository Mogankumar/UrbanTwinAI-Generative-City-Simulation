import folium
import numpy as np
import branca.colormap as cm

def add_heat_layer(m, grid_gdf, values, name, vmin=None, vmax=None):
    g = grid_gdf.copy()
    if getattr(g, "crs", None) is not None:
        try:
            g = g.to_crs(epsg=4326)
        except Exception:
            pass

    v = np.asarray(values, dtype=float)
    if vmin is None: vmin = -0.8 if "UHI" in name else -0.2  
    if vmax is None: vmax =  0.8 if "UHI" in name else  0.2
    if np.isclose(vmin, vmax):
        vmax = vmin + 1e-6  

    col = cm.LinearColormap(["blue", "yellow", "red"], vmin=vmin, vmax=vmax)

    gj = g.assign(val=v).to_json()
    folium.GeoJson(
        gj,
        name=name,
        style_function=lambda feat: {
            "fillColor": col(feat["properties"]["val"]),
            "color": "#333333",           
            "weight": 0.4,
            "fillOpacity": 0.6
        },
        tooltip=folium.GeoJsonTooltip(fields=["val"], aliases=[name]),
    ).add_to(m)

    col.caption = name
    col.add_to(m)
    return m


def add_road_layer(m, roads_val_gdf, name, vmin=None, vmax=None, weight=3):
    v = roads_val_gdf["val"].values
    vmin = float(np.nanmin(v) if vmin is None else vmin)
    vmax = float(np.nanmax(v) if vmax is None else vmax)
    col = cm.LinearColormap(["blue", "yellow", "red"], vmin=vmin, vmax=vmax)

    gj = folium.GeoJson(
        roads_val_gdf[["geometry", "val"]].to_json(),
        name=name,
        style_function=lambda feat: {
            "color": col(feat["properties"]["val"]),
            "weight": weight,
            "opacity": 0.9,
        },
        highlight_function=lambda feat: {"weight": weight + 2}
    )
    gj.add_to(m)
    col.caption = name
    col.add_to(m)
    return m

def add_heat_layer(
    m, grid_gdf, values, legend,
    vmin=None, vmax=None,
    cmap="YlOrRd_09",
    fill_opacity=0.85,
    line_opacity=0.15,
    line_weight=0.2,
):
    g = grid_gdf.copy()
    g["__val__"] = np.asarray(values, dtype=float)

    try:
        if g.crs is not None:
            g = g.to_crs(epsg=4326)         
    except Exception:
        pass

    if vmin is None: vmin = float(np.nanmin(g["__val__"]))
    if vmax is None: vmax = float(np.nanmax(g["__val__"]))
    if not np.isfinite(vmin): vmin = 0.0
    if not np.isfinite(vmax) or vmax == vmin: vmax = vmin + 1e-9

    try:
        base = getattr(cm.linear, cmap) 
        colormap = base.scale(vmin, vmax)
    except Exception:
        colormap = cm.LinearColormap(
            colors=["#ffffcc", "#fd8d3c", "#800026"], vmin=vmin, vmax=vmax
        )

    def style_fn(feat):
        v = feat["properties"]["__val__"]
        color = colormap(v)
        return {
            "fillColor": color,
            "color": color,
            "weight": line_weight,
            "fillOpacity": fill_opacity,
            "opacity": line_opacity,
        }

    folium.GeoJson(
        g.to_json(),
        name=legend,
        style_function=style_fn,
        tooltip=folium.GeoJsonTooltip(fields=["__val__"], aliases=[legend]),
    ).add_to(m)

    colormap.caption = legend
    colormap.add_to(m)
