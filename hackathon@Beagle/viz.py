import folium
import numpy as np
import branca.colormap as cm

def add_heat_layer(m, grid_gdf, values, name, vmin=None, vmax=None):
    v = values.values
    vmin = vmin if vmin is not None else float(np.percentile(v, 5))
    vmax = vmax if vmax is not None else float(np.percentile(v, 95))
    col = cm.LinearColormap(["blue","yellow","red"], vmin=vmin, vmax=vmax)

    gj = grid_gdf.assign(val=values.values).to_json()
    folium.GeoJson(
        gj,
        name=name,
        style_function=lambda feat: {"fillColor": col(feat["properties"]["val"]),
                                     "color": None, "fillOpacity": 0.6},
    ).add_to(m)
    col.caption = name
    col.add_to(m)
    return m