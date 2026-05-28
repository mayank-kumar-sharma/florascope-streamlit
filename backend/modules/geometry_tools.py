"""
Geometry & GIS Utility Tools
Handles: area calculations, format conversions, KML/GPKG parsing,
polygon operations, coordinate transforms, and KML export.
"""

import json
import math
import os
import tempfile
import uuid
import zipfile
from typing import Dict, List, Optional, Tuple, Union

import fiona
from shapely.geometry import shape, mapping, Polygon, MultiPolygon
from shapely.ops import unary_union, transform
import pyproj
from functools import partial


# ──────────────────────────────────────────────
# COORDINATE REFERENCE SYSTEMS
# ──────────────────────────────────────────────

def get_utm_crs(lon: float, lat: float) -> pyproj.CRS:
    """Get the appropriate UTM CRS for a given lon/lat."""
    zone = int((lon + 180) / 6) + 1
    hemisphere = "north" if lat >= 0 else "south"
    epsg = 32600 + zone if hemisphere == "north" else 32700 + zone
    return pyproj.CRS.from_epsg(epsg)


def reproject_geometry(geom, src_crs="EPSG:4326", dst_crs=None, lon=None, lat=None):
    """Reproject a Shapely geometry between CRS."""
    if dst_crs is None and lon is not None:
        dst_crs = get_utm_crs(lon, lat)
    elif dst_crs is None:
        centroid = geom.centroid
        dst_crs = get_utm_crs(centroid.x, centroid.y)

    project = pyproj.Transformer.from_crs(
        pyproj.CRS(src_crs), dst_crs, always_xy=True
    ).transform
    return transform(project, geom)


# ──────────────────────────────────────────────
# AREA CALCULATIONS
# ──────────────────────────────────────────────

def calculate_area(geojson_data: dict) -> Dict:
    """
    Calculate area of a GeoJSON geometry in multiple units.
    Uses UTM projection for accurate area calculation.
    """
    geom = _extract_geometry(geojson_data)
    if geom is None or geom.is_empty:
        return {"error": "Invalid or empty geometry"}

    # Fix invalid geometry
    if not geom.is_valid:
        geom = geom.buffer(0)

    # Reproject to UTM for accurate area
    centroid = geom.centroid
    geom_utm = reproject_geometry(geom, lon=centroid.x, lat=centroid.y)

    area_sq_m = geom_utm.area
    perimeter_m = geom_utm.length

    return {
        "area_sq_m": round(area_sq_m, 2),
        "area_sq_ft": round(area_sq_m * 10.7639, 2),
        "area_sq_km": round(area_sq_m / 1_000_000, 4),
        "area_hectares": round(area_sq_m / 10_000, 4),
        "area_acres": round(area_sq_m / 4046.86, 4),
        "area_sq_miles": round(area_sq_m / 2_589_988, 6),
        "perimeter_m": round(perimeter_m, 2),
        "perimeter_km": round(perimeter_m / 1000, 4),
        "perimeter_ft": round(perimeter_m * 3.28084, 2),
        "centroid_lat": round(centroid.y, 6),
        "centroid_lon": round(centroid.x, 6),
        "bounds": {
            "min_lon": round(geom.bounds[0], 6),
            "min_lat": round(geom.bounds[1], 6),
            "max_lon": round(geom.bounds[2], 6),
            "max_lat": round(geom.bounds[3], 6),
        },
        "num_vertices": _count_vertices(geom),
        "geometry_type": geom.geom_type,
    }


def calculate_multi_area(geojson_features: List[dict]) -> Dict:
    """Calculate area for multiple features and return individual + total."""
    results = []
    total_area_ha = 0

    for i, feat in enumerate(geojson_features):
        area = calculate_area(feat)
        area["feature_index"] = i
        area["name"] = feat.get("properties", {}).get("name", f"Feature {i+1}")
        results.append(area)
        total_area_ha += area.get("area_hectares", 0)

    return {
        "features": results,
        "total_area_hectares": round(total_area_ha, 4),
        "total_area_sq_m": round(total_area_ha * 10000, 2),
        "total_area_acres": round(total_area_ha * 2.47105, 4),
        "feature_count": len(results),
    }


# ──────────────────────────────────────────────
# FILE PARSING (KML, GPKG, GeoJSON, SHP)
# ──────────────────────────────────────────────

def parse_upload(file_path: str, file_extension: str) -> dict:
    """
    Universal file parser. Returns GeoJSON FeatureCollection.
    Supports: .geojson, .json, .kml, .kmz, .gpkg, .shp (in .zip)
    """
    ext = file_extension.lower()

    if ext in (".geojson", ".json"):
        with open(file_path, "r") as f:
            data = json.load(f)
        return _normalize_geojson(data)

    elif ext == ".kml":
        return _parse_kml(file_path)

    elif ext == ".kmz":
        return _parse_kmz(file_path)

    elif ext == ".gpkg":
        return _parse_gpkg(file_path)

    elif ext == ".zip":
        return _parse_shapefile_zip(file_path)

    elif ext == ".shp":
        return _parse_fiona(file_path)

    else:
        raise ValueError(f"Unsupported file format: {ext}")


def _parse_kml(file_path: str) -> dict:
    """Parse KML file using fiona."""
    fiona.drvsupport.supported_drivers["KML"] = "rw"
    fiona.drvsupport.supported_drivers["LIBKML"] = "rw"

    features = []
    with fiona.open(file_path, driver="KML") as src:
        for feat in src:
            geom = shape(feat["geometry"])
            if not geom.is_valid:
                geom = geom.buffer(0)
            props = dict(feat.get("properties", {}))
            # KML Name/Description
            name = props.get("Name", props.get("name", ""))
            features.append({
                "type": "Feature",
                "geometry": mapping(geom),
                "properties": {"name": name, **props}
            })

    return {"type": "FeatureCollection", "features": features}


def _parse_kmz(file_path: str) -> dict:
    """Parse KMZ (zipped KML) file."""
    extract_dir = tempfile.mkdtemp()
    try:
        with zipfile.ZipFile(file_path, "r") as z:
            z.extractall(extract_dir)

        # Find .kml inside
        for root, dirs, files in os.walk(extract_dir):
            for f in files:
                if f.lower().endswith(".kml"):
                    return _parse_kml(os.path.join(root, f))

        raise ValueError("No .kml file found inside .kmz archive")
    finally:
        import shutil
        shutil.rmtree(extract_dir, ignore_errors=True)


def _parse_gpkg(file_path: str) -> dict:
    """Parse GeoPackage file."""
    features = []
    # List layers
    layers = fiona.listlayers(file_path)

    for layer_name in layers:
        with fiona.open(file_path, layer=layer_name) as src:
            for feat in src:
                geom = shape(feat["geometry"])
                if not geom.is_valid:
                    geom = geom.buffer(0)
                props = dict(feat.get("properties", {}))
                props["_layer"] = layer_name
                features.append({
                    "type": "Feature",
                    "geometry": mapping(geom),
                    "properties": props
                })

    return {"type": "FeatureCollection", "features": features}


def _parse_shapefile_zip(file_path: str) -> dict:
    """Parse shapefile from zip archive."""
    extract_dir = tempfile.mkdtemp()
    try:
        with zipfile.ZipFile(file_path, "r") as z:
            z.extractall(extract_dir)

        # Find .shp file
        for root, dirs, files in os.walk(extract_dir):
            for f in files:
                if f.lower().endswith(".shp"):
                    return _parse_fiona(os.path.join(root, f))

        raise ValueError("No .shp file found in zip archive")
    finally:
        import shutil
        shutil.rmtree(extract_dir, ignore_errors=True)


def _parse_fiona(file_path: str) -> dict:
    """Generic fiona-based parser."""
    features = []
    with fiona.open(file_path) as src:
        for feat in src:
            geom = shape(feat["geometry"])
            if not geom.is_valid:
                geom = geom.buffer(0)
            features.append({
                "type": "Feature",
                "geometry": mapping(geom),
                "properties": dict(feat.get("properties", {}))
            })
    return {"type": "FeatureCollection", "features": features}


def _normalize_geojson(data: dict) -> dict:
    """Normalize various GeoJSON inputs to FeatureCollection."""
    if data.get("type") == "FeatureCollection":
        return data
    elif data.get("type") == "Feature":
        return {"type": "FeatureCollection", "features": [data]}
    elif data.get("type") in ("Polygon", "MultiPolygon", "Point", "LineString"):
        return {
            "type": "FeatureCollection",
            "features": [{
                "type": "Feature",
                "geometry": data,
                "properties": {}
            }]
        }
    raise ValueError(f"Unrecognized GeoJSON type: {data.get('type')}")


# ──────────────────────────────────────────────
# KML EXPORT
# ──────────────────────────────────────────────

def export_to_kml(geojson_data: dict, name: str = "Export") -> str:
    """Convert GeoJSON to KML string."""
    features = geojson_data.get("features", [])
    if not features:
        if geojson_data.get("type") == "Feature":
            features = [geojson_data]

    placemarks = []
    for i, feat in enumerate(features):
        feat_name = feat.get("properties", {}).get("name", f"Feature {i+1}")
        geom = feat.get("geometry", {})
        coords_kml = _geometry_to_kml_coords(geom)

        placemarks.append(f"""
    <Placemark>
      <name>{_xml_escape(feat_name)}</name>
      <description>{_xml_escape(json.dumps(feat.get("properties", {})))}</description>
      <Style>
        <LineStyle><color>ff00aa00</color><width>2</width></LineStyle>
        <PolyStyle><color>4d00ff00</color></PolyStyle>
      </Style>
      {coords_kml}
    </Placemark>""")

    kml = f"""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Document>
    <name>{_xml_escape(name)}</name>
    {"".join(placemarks)}
  </Document>
</kml>"""
    return kml


def export_to_geojson(geojson_data: dict) -> str:
    """Clean export of GeoJSON."""
    return json.dumps(geojson_data, indent=2)


def _geometry_to_kml_coords(geom: dict) -> str:
    """Convert GeoJSON geometry to KML coordinate elements."""
    gtype = geom.get("type", "")
    coords = geom.get("coordinates", [])

    if gtype == "Polygon":
        rings = []
        for ring in coords:
            coord_str = " ".join(f"{c[0]},{c[1]},0" for c in ring)
            rings.append(f"<LinearRing><coordinates>{coord_str}</coordinates></LinearRing>")
        outer = f"<outerBoundaryIs>{rings[0]}</outerBoundaryIs>" if rings else ""
        inner = "".join(f"<innerBoundaryIs>{r}</innerBoundaryIs>" for r in rings[1:])
        return f"<Polygon>{outer}{inner}</Polygon>"

    elif gtype == "MultiPolygon":
        polygons = []
        for poly_coords in coords:
            rings = []
            for ring in poly_coords:
                coord_str = " ".join(f"{c[0]},{c[1]},0" for c in ring)
                rings.append(f"<LinearRing><coordinates>{coord_str}</coordinates></LinearRing>")
            outer = f"<outerBoundaryIs>{rings[0]}</outerBoundaryIs>" if rings else ""
            inner = "".join(f"<innerBoundaryIs>{r}</innerBoundaryIs>" for r in rings[1:])
            polygons.append(f"<Polygon>{outer}{inner}</Polygon>")
        return f"<MultiGeometry>{''.join(polygons)}</MultiGeometry>"

    elif gtype == "Point":
        return f"<Point><coordinates>{coords[0]},{coords[1]},0</coordinates></Point>"

    elif gtype == "LineString":
        coord_str = " ".join(f"{c[0]},{c[1]},0" for c in coords)
        return f"<LineString><coordinates>{coord_str}</coordinates></LineString>"

    return ""


def _xml_escape(s: str) -> str:
    """Escape XML special characters."""
    return (str(s).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;").replace("'", "&apos;"))


def _count_vertices(geom) -> int:
    """Count vertices in a geometry."""
    if hasattr(geom, "exterior"):
        return len(geom.exterior.coords)
    elif hasattr(geom, "geoms"):
        return sum(_count_vertices(g) for g in geom.geoms)
    elif hasattr(geom, "coords"):
        return len(geom.coords)
    return 0


def _extract_geometry(data: dict):
    """Extract Shapely geometry from various GeoJSON inputs."""
    if data.get("type") == "FeatureCollection":
        features = data.get("features", [])
        if not features:
            return None
        geoms = [shape(f["geometry"]) for f in features if f.get("geometry")]
        return unary_union(geoms) if geoms else None
    elif data.get("type") == "Feature":
        return shape(data["geometry"])
    elif data.get("type") in ("Polygon", "MultiPolygon", "Point", "LineString"):
        return shape(data)
    return None


# ──────────────────────────────────────────────
# BUFFER / SIMPLIFY / OPERATIONS
# ──────────────────────────────────────────────

def buffer_geometry(geojson_data: dict, distance_m: float) -> dict:
    """Buffer a geometry by distance in meters. Returns GeoJSON."""
    geom = _extract_geometry(geojson_data)
    centroid = geom.centroid
    geom_utm = reproject_geometry(geom, lon=centroid.x, lat=centroid.y)
    buffered_utm = geom_utm.buffer(distance_m)

    # Reproject back to WGS84
    utm_crs = get_utm_crs(centroid.x, centroid.y)
    project_back = pyproj.Transformer.from_crs(
        utm_crs, pyproj.CRS("EPSG:4326"), always_xy=True
    ).transform
    buffered_wgs = transform(project_back, buffered_utm)

    return {
        "type": "Feature",
        "geometry": mapping(buffered_wgs),
        "properties": {"buffer_distance_m": distance_m}
    }


def simplify_geometry(geojson_data: dict, tolerance: float = 0.0001) -> dict:
    """Simplify geometry to reduce vertices."""
    geom = _extract_geometry(geojson_data)
    simplified = geom.simplify(tolerance, preserve_topology=True)
    return {
        "type": "Feature",
        "geometry": mapping(simplified),
        "properties": {
            "original_vertices": _count_vertices(geom),
            "simplified_vertices": _count_vertices(simplified),
        }
    }
