import geopandas as gpd
import numpy as np
import osmnx as ox
from shapely.geometry import box
import pandas as pd


def _only_polygons(gdf):
    return gdf[gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()

def _only_lines(gdf):
    return gdf[gdf.geometry.geom_type.isin(["LineString", "MultiLineString"])].copy()

def _fix_invalid_polys(gdf):
    # buffer(0) fixes many invalid polygon issues
    gdf = gdf.copy()
    gdf["geometry"] = gdf.buffer(0)
    return gdf[~gdf.geometry.is_empty]

def make_bbox(center_lat, center_lon, km=1.0):
    d = km * 1000
    dy = d / 111_320
    dx = d / (40075000 * np.cos(np.radians(center_lat)) / 360)
    return (center_lon - dx/2, center_lat - dy/2, center_lon + dx/2, center_lat + dy/2)

def fetch_osm(bbox):
    west, south, east, north = bbox
    polygon = box(west, south, east, north)

    # OSMnx v2+ API
    bld = ox.features_from_polygon(polygon, tags={"building": True})
    roads_graph = ox.graph_from_polygon(polygon, network_type="drive")
    roads = ox.graph_to_gdfs(roads_graph, nodes=False, edges=True)
    green = ox.features_from_polygon(
        polygon,
        tags={"landuse": ["grass","forest","meadow","park"], "leisure": ["park","garden"]}
    )

    # project to metric CRS
    crs_m = "EPSG:3857"
    bld   = bld.to_crs(crs_m)
    roads = roads.to_crs(crs_m)
    green = green.to_crs(crs_m)

    # ✅ keep only valid types for our ops
    bld   = _only_polygons(bld)
    green = _only_polygons(green)
    roads = _only_lines(roads)

    # ✅ repair invalid polygons (common in OSM)
    bld   = _fix_invalid_polys(bld)
    green = _fix_invalid_polys(green)

    poly_m = gpd.GeoSeries([polygon], crs="EPSG:4326").to_crs(crs_m).iloc[0]
    return bld, roads, green, poly_m

def grid_bbox(polygon_m, cell=50):
    minx, miny, maxx, maxy = polygon_m.bounds
    xs = np.arange(minx, maxx, cell)
    ys = np.arange(miny, maxy, cell)
    cells = [box(x, y, x+cell, y+cell) for x in xs for y in ys]
    return gpd.GeoDataFrame(geometry=cells, crs="EPSG:3857")

def features(grid, bld, roads, green):
    # work on copies and give every grid cell a stable id
    g = grid.copy()
    g["cell_area"] = g.geometry.area
    g2 = g[["geometry"]].reset_index().rename(columns={"index": "grid_id"})

    # --- Buildings: coverage (area fraction) ---
    b2 = bld[["geometry"]].reset_index().rename(columns={"index": "bld_id"})
    inter_b = gpd.overlay(g2, b2, how="intersection", keep_geom_type=False)
    if inter_b.empty:
        b_area = pd.Series(0, index=g2["grid_id"])
    else:
        b_area = inter_b.geometry.area.groupby(inter_b["grid_id"]).sum()
    b_area = b_area.reindex(g2["grid_id"]).fillna(0)
    g["building_cov"] = (b_area.values / g["cell_area"]).clip(0, 1)

    # --- Greenspace: coverage (area fraction) ---
    gr2 = green[["geometry"]].reset_index().rename(columns={"index": "green_id"})
    inter_g = gpd.overlay(g2, gr2, how="intersection", keep_geom_type=False)
    if inter_g.empty:
        g_area = pd.Series(0, index=g2["grid_id"])
    else:
        g_area = inter_g.geometry.area.groupby(inter_g["grid_id"]).sum()
    g_area = g_area.reindex(g2["grid_id"]).fillna(0)
    g["green_cov"] = (g_area.values / g["cell_area"]).clip(0, 1)

    # --- Roads: density (length per m²) ---
    r2 = roads[["geometry"]].reset_index().rename(columns={"index": "road_id"})

    # Replace the old overlay block with this:
    r_clipped = gpd.clip(r2, g2)  # keep only road pieces inside grid cells
    if r_clipped.empty:
        r_len = pd.Series(0, index=g2["grid_id"])
    else:
        # tag each clipped segment with its grid_id
        r_clipped = gpd.sjoin(r_clipped, g2, predicate="intersects", how="left")
        r_len = r_clipped.length.groupby(r_clipped["grid_id"]).sum()

    r_len = r_len.reindex(g2["grid_id"]).fillna(0)
    g["road_den"] = (r_len.values / g["cell_area"])

    # simple extras
    g["int_den"] = g["road_den"] * 0.8
    g["impervious"] = (g["building_cov"] + g["road_den"] * 0.02).clip(0, 1)
    return g