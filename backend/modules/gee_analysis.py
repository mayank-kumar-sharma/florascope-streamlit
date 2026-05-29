from __future__ import annotations

"""
Google Earth Engine Integration Module
Handles LULC classification, Hansen forest data, and mangrove analysis.
"""

import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple

try:
    import ee
except ImportError:
    ee = None

try:
    import numpy as np
except ImportError:
    np = None


# ──────────────────────────────────────────────
# ESRI 10m LULC Class Mapping
# ──────────────────────────────────────────────
ESRI_LULC_CLASSES = {
    1: {"name": "No Data", "color": "#ffffff"},
    2: {"name": "Built Area", "color": "#d13c1b"},
    3: {"name": "Crops", "color": "#e8d354"},
    4: {"name": "Bare Ground", "color": "#a39171"},
    5: {"name": "Snow/Ice", "color": "#b5e7ff"},
    6: {"name": "Clouds", "color": "#c9c9c9"},
    7: {"name": "Rangeland", "color": "#a0c93d"},  # Shrub/Grassland
    8: {"name": "Flooded Vegetation", "color": "#7a87c6"},
    9: {"name": "Water", "color": "#0096ff"},
    10: {"name": "Trees", "color": "#1a9850"},
    11: {"name": "Mangroves", "color": "#006400"},
}

# Dynamic World class mapping
DYNAMIC_WORLD_CLASSES = {
    0: {"name": "Water", "color": "#419BDF"},
    1: {"name": "Trees", "color": "#397D49"},
    2: {"name": "Grass", "color": "#88B053"},
    3: {"name": "Flooded Vegetation", "color": "#7A87C6"},
    4: {"name": "Crops", "color": "#E49635"},
    5: {"name": "Shrub & Scrub", "color": "#DFC35A"},
    6: {"name": "Built Area", "color": "#C4281B"},
    7: {"name": "Bare Ground", "color": "#A59B8F"},
    8: {"name": "Snow & Ice", "color": "#B39FE1"},
}

# Hansen Global Forest Change thresholds
HANSEN_TREE_COVER_THRESHOLD = 30  # % canopy cover to be classified as forest


def initialize_gee(service_account_key: Optional[str] = None):
    """Initialize Google Earth Engine with service account or default credentials."""
    try:
        if service_account_key:
            credentials = ee.ServiceAccountCredentials(
                email=json.loads(service_account_key)["client_email"],
                key_data=service_account_key
            )
            ee.Initialize(credentials)
        else:
            ee.Authenticate()
            ee.Initialize(project="ee-project")
        return True
    except Exception as e:
        print(f"GEE initialization error: {e}")
        return False


def geometry_from_geojson(geojson_data: dict) -> ee.Geometry:
    """Convert GeoJSON to EE Geometry."""
    if geojson_data.get("type") == "FeatureCollection":
        features = geojson_data["features"]
        if features:
            return ee.Geometry(features[0]["geometry"])
    elif geojson_data.get("type") == "Feature":
        return ee.Geometry(geojson_data["geometry"])
    else:
        return ee.Geometry(geojson_data)


# ──────────────────────────────────────────────
# LULC ANALYSIS (10-Year Time Series)
# ──────────────────────────────────────────────

def get_esri_lulc_timeseries(aoi: ee.Geometry, start_year: int, end_year: int) -> Dict:
    """
    Get ESRI 10m Annual LULC for each year in the range.
    Returns pixel counts per class per year.
    """
    results = {}
    for year in range(start_year, end_year + 1):
        try:
            lulc = ee.ImageCollection(
                "projects/sat-io/open-datasets/land-cover/ESRI_Global-LULC_10m_TS"
            ).filterDate(f"{year}-01-01", f"{year}-12-31").first()

            if lulc is None:
                continue

            clipped = lulc.clip(aoi)

            # Count pixels per class
            area_image = ee.Image.pixelArea().addBands(clipped)
            stats = area_image.reduceRegion(
                reducer=ee.Reducer.sum().group(
                    groupField=1,
                    groupName="class"
                ),
                geometry=aoi,
                scale=10,
                maxPixels=1e10,
                bestEffort=True
            )

            groups = stats.get("groups").getInfo()
            year_data = {}
            if groups:
                for g in groups:
                    cls = int(g["class"])
                    area_m2 = g["sum"]
                    area_ha = area_m2 / 10000
                    if cls in ESRI_LULC_CLASSES:
                        year_data[ESRI_LULC_CLASSES[cls]["name"]] = round(area_ha, 2)
            results[year] = year_data
        except Exception as e:
            print(f"Error for year {year}: {e}")
            results[year] = {}

    return results


def get_dynamic_world_lulc(aoi: ee.Geometry, start_year: int, end_year: int) -> Dict:
    """
    Alternative: Use Google Dynamic World 10m LULC.
    Provides near-real-time land cover classification.
    """
    results = {}
    for year in range(start_year, end_year + 1):
        try:
            dw = ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1") \
                .filterBounds(aoi) \
                .filterDate(f"{year}-01-01", f"{year}-12-31")

            # Get the mode (most common) classification
            mode_image = dw.select("label").mode().clip(aoi)

            area_image = ee.Image.pixelArea().addBands(mode_image)
            stats = area_image.reduceRegion(
                reducer=ee.Reducer.sum().group(groupField=1, groupName="class"),
                geometry=aoi,
                scale=10,
                maxPixels=1e10,
                bestEffort=True
            )

            groups = stats.get("groups").getInfo()
            year_data = {}
            if groups:
                for g in groups:
                    cls = int(g["class"])
                    area_ha = g["sum"] / 10000
                    if cls in DYNAMIC_WORLD_CLASSES:
                        year_data[DYNAMIC_WORLD_CLASSES[cls]["name"]] = round(area_ha, 2)
            results[year] = year_data
        except Exception as e:
            print(f"Dynamic World error for year {year}: {e}")
            results[year] = {}

    return results


def get_lulc_map_tiles(aoi: ee.Geometry, year: int) -> str:
    """Generate a tile URL for LULC visualization."""
    try:
        lulc = ee.ImageCollection(
            "projects/sat-io/open-datasets/land-cover/ESRI_Global-LULC_10m_TS"
        ).filterDate(f"{year}-01-01", f"{year}-12-31").first()

        vis_params = {
            "min": 1, "max": 11,
            "palette": [
                "#ffffff", "#d13c1b", "#e8d354", "#a39171", "#b5e7ff",
                "#c9c9c9", "#a0c93d", "#7a87c6", "#0096ff", "#1a9850", "#006400"
            ]
        }

        map_id = lulc.clip(aoi).getMapId(vis_params)
        return map_id["tile_fetcher"].url_format
    except Exception as e:
        print(f"Tile generation error: {e}")
        return None


# ──────────────────────────────────────────────
# HANSEN GLOBAL FOREST CHANGE
# ──────────────────────────────────────────────

def analyze_hansen_forest(aoi: ee.Geometry) -> Dict:
    """
    Analyze Hansen Global Forest Change v1.11 dataset.
    Returns: tree cover 2000, gain, loss by year, net change.
    """
    hansen = ee.Image("UMD/hansen/global_forest_change_2023_v1_11")

    # Tree cover in 2000
    tree_cover_2000 = hansen.select("treecover2000").clip(aoi)

    # Forest mask (>30% canopy)
    forest_2000 = tree_cover_2000.gte(HANSEN_TREE_COVER_THRESHOLD)

    # Forest gain (2000-2012)
    gain = hansen.select("gain").clip(aoi)

    # Loss year (1-23 = 2001-2023)
    loss_year = hansen.select("lossyear").clip(aoi)

    # Total loss mask
    loss = hansen.select("loss").clip(aoi)

    # Calculate areas
    pixel_area = ee.Image.pixelArea()

    # Forest area in 2000
    forest_area_2000 = pixel_area.updateMask(forest_2000).reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=aoi,
        scale=30,
        maxPixels=1e10,
        bestEffort=True
    ).get("area")

    # Total area
    total_area = pixel_area.reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=aoi,
        scale=30,
        maxPixels=1e10,
        bestEffort=True
    ).get("area")

    # Forest gain area
    gain_area = pixel_area.updateMask(gain).reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=aoi,
        scale=30,
        maxPixels=1e10,
        bestEffort=True
    ).get("area")

    # Forest loss area
    loss_area = pixel_area.updateMask(loss).reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=aoi,
        scale=30,
        maxPixels=1e10,
        bestEffort=True
    ).get("area")

    # Loss by year
    loss_by_year = {}
    for yr in range(1, 24):  # 2001-2023
        yr_mask = loss_year.eq(yr)
        yr_area = pixel_area.updateMask(yr_mask).reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=aoi,
            scale=30,
            maxPixels=1e10,
            bestEffort=True
        ).get("area")
        loss_by_year[2000 + yr] = round((yr_area.getInfo() or 0) / 10000, 2)

    # Mean tree cover percentage
    mean_cover = tree_cover_2000.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=aoi,
        scale=30,
        maxPixels=1e10,
        bestEffort=True
    ).get("treecover2000")

    return {
        "total_area_ha": round((total_area.getInfo() or 0) / 10000, 2),
        "forest_area_2000_ha": round((forest_area_2000.getInfo() or 0) / 10000, 2),
        "forest_gain_ha": round((gain_area.getInfo() or 0) / 10000, 2),
        "forest_loss_total_ha": round((loss_area.getInfo() or 0) / 10000, 2),
        "loss_by_year": loss_by_year,
        "mean_tree_cover_2000_pct": round(mean_cover.getInfo() or 0, 1),
        "forest_cover_pct_2000": round(
            ((forest_area_2000.getInfo() or 0) / max((total_area.getInfo() or 1), 1)) * 100, 1
        ),
    }


def get_hansen_map_tiles(aoi: ee.Geometry) -> Dict[str, str]:
    """Generate tile URLs for Hansen visualization layers."""
    hansen = ee.Image("UMD/hansen/global_forest_change_2023_v1_11")
    tiles = {}

    try:
        # Tree cover 2000
        tc = hansen.select("treecover2000").clip(aoi)
        tc_vis = tc.getMapId({"min": 0, "max": 100, "palette": ["#ffffcc", "#006400"]})
        tiles["tree_cover_2000"] = tc_vis["tile_fetcher"].url_format

        # Loss (red)
        loss = hansen.select("loss").clip(aoi).selfMask()
        loss_vis = loss.getMapId({"palette": ["#FF0000"]})
        tiles["forest_loss"] = loss_vis["tile_fetcher"].url_format

        # Gain (blue)
        gain = hansen.select("gain").clip(aoi).selfMask()
        gain_vis = gain.getMapId({"palette": ["#0000FF"]})
        tiles["forest_gain"] = gain_vis["tile_fetcher"].url_format
    except Exception as e:
        print(f"Hansen tile error: {e}")

    return tiles


# ──────────────────────────────────────────────
# MANGROVE ANALYSIS
# ──────────────────────────────────────────────

def analyze_mangroves(aoi: ee.Geometry) -> Dict:
    """
    Analyze Global Mangrove Watch (GMW) data for mangrove presence and change.
    Also checks JRC Global Surface Water for tidal/wetland conditions.
    """
    result = {
        "mangrove_present": False,
        "mangrove_area_ha": 0,
        "mangrove_loss_ha": 0,
        "tidal_wetland": False,
        "historical_mangrove": False,
        "mangrove_years": {},
    }

    try:
        # Global Mangrove Watch
        gmw = ee.ImageCollection("projects/earthengine-legacy/assets/projects/sat-io/open-datasets/GMW/union")
        pixel_area = ee.Image.pixelArea()

        # Check multiple years if available
        for year in [1996, 2007, 2008, 2009, 2010, 2015, 2016, 2017, 2018, 2019, 2020]:
            try:
                mangrove_img = gmw.filterDate(f"{year}-01-01", f"{year}-12-31").first()
                if mangrove_img:
                    clipped = mangrove_img.clip(aoi).selfMask()
                    area = pixel_area.updateMask(clipped).reduceRegion(
                        reducer=ee.Reducer.sum(),
                        geometry=aoi,
                        scale=30,
                        maxPixels=1e10,
                        bestEffort=True
                    )
                    area_val = area.getInfo()
                    if area_val:
                        area_ha = list(area_val.values())[0] / 10000 if area_val else 0
                        result["mangrove_years"][year] = round(area_ha, 2)
                        if area_ha > 0:
                            result["historical_mangrove"] = True
            except:
                pass

        # Current mangrove from latest available year
        latest_years = sorted(result["mangrove_years"].keys(), reverse=True)
        if latest_years:
            result["mangrove_area_ha"] = result["mangrove_years"][latest_years[0]]
            result["mangrove_present"] = result["mangrove_area_ha"] > 0

        # Calculate loss if historical data available
        if len(result["mangrove_years"]) >= 2:
            earliest = result["mangrove_years"][min(result["mangrove_years"].keys())]
            latest = result["mangrove_years"][max(result["mangrove_years"].keys())]
            result["mangrove_loss_ha"] = round(max(0, earliest - latest), 2)

    except Exception as e:
        print(f"Mangrove analysis error: {e}")

    # Check for tidal/wetland conditions using JRC Surface Water
    try:
        jrc = ee.Image("JRC/GSW1_4/GlobalSurfaceWater")
        occurrence = jrc.select("occurrence").clip(aoi)
        seasonality = jrc.select("seasonality").clip(aoi)

        # Areas with seasonal water (potential tidal influence)
        seasonal_water = seasonality.gt(0).And(seasonality.lt(12))
        seasonal_area = ee.Image.pixelArea().updateMask(seasonal_water).reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=aoi,
            scale=30,
            maxPixels=1e10,
            bestEffort=True
        )
        seasonal_val = seasonal_area.getInfo()
        if seasonal_val:
            seasonal_ha = list(seasonal_val.values())[0] / 10000 if seasonal_val else 0
            result["tidal_wetland"] = seasonal_ha > 0.5  # >0.5 ha seasonal water

    except Exception as e:
        print(f"Tidal analysis error: {e}")

    return result


# ──────────────────────────────────────────────
# NDVI TIME SERIES (Vegetation Health)
# ──────────────────────────────────────────────

def get_ndvi_timeseries(aoi: ee.Geometry, start_year: int, end_year: int) -> Dict:
    """
    Get annual mean NDVI from Sentinel-2 or Landsat for vegetation health analysis.
    """
    results = {}
    for year in range(start_year, end_year + 1):
        try:
            if year >= 2017:
                # Use Sentinel-2
                collection = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED") \
                    .filterBounds(aoi) \
                    .filterDate(f"{year}-01-01", f"{year}-12-31") \
                    .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))

                def calc_ndvi_s2(img):
                    return img.normalizedDifference(["B8", "B4"]).rename("NDVI")

                ndvi = collection.map(calc_ndvi_s2).mean().clip(aoi)
            else:
                # Use Landsat 8
                collection = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2") \
                    .filterBounds(aoi) \
                    .filterDate(f"{year}-01-01", f"{year}-12-31") \
                    .filter(ee.Filter.lt("CLOUD_COVER", 20))

                def calc_ndvi_l8(img):
                    return img.normalizedDifference(["SR_B5", "SR_B4"]).rename("NDVI")

                ndvi = collection.map(calc_ndvi_l8).mean().clip(aoi)

            mean_ndvi = ndvi.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=aoi,
                scale=30,
                maxPixels=1e10,
                bestEffort=True
            ).get("NDVI")

            results[year] = round(mean_ndvi.getInfo() or 0, 3)
        except Exception as e:
            print(f"NDVI error for year {year}: {e}")
            results[year] = None

    return results


# ──────────────────────────────────────────────
# SOIL ORGANIC CARBON
# ──────────────────────────────────────────────

def get_soil_carbon(aoi: ee.Geometry) -> Dict:
    """
    Get soil organic carbon data from SoilGrids.
    Useful for baseline carbon stock estimation.
    """
    try:
        soc = ee.Image("projects/soilgrids-isric/ocd_mean") \
            .select("ocd_0-5cm_mean").clip(aoi)

        stats = soc.reduceRegion(
            reducer=ee.Reducer.mean().combine(
                ee.Reducer.min(), sharedInputs=True
            ).combine(
                ee.Reducer.max(), sharedInputs=True
            ),
            geometry=aoi,
            scale=250,
            maxPixels=1e10,
            bestEffort=True
        )

        info = stats.getInfo() if stats is not None else None
        if info is None:
            return {"mean_soc_g_per_kg": 0, "min_soc_g_per_kg": 0, "max_soc_g_per_kg": 0}
        
        # Safely extract and divide values, checking for None before division
        mean_val = info.get("ocd_0-5cm_mean_mean")
        mean_soc = round(mean_val / 10, 1) if mean_val is not None else 0
        
        min_val = info.get("ocd_0-5cm_mean_min")
        min_soc = round(min_val / 10, 1) if min_val is not None else 0
        
        max_val = info.get("ocd_0-5cm_mean_max")
        max_soc = round(max_val / 10, 1) if max_val is not None else 0
        
        return {
            "mean_soc_g_per_kg": mean_soc,
            "min_soc_g_per_kg": min_soc,
            "max_soc_g_per_kg": max_soc,
        }
    except Exception as e:
        print(f"SOC analysis error: {e}")
        return {"mean_soc_g_per_kg": 0, "min_soc_g_per_kg": 0, "max_soc_g_per_kg": 0}


# ──────────────────────────────────────────────
# ELEVATION / SLOPE ANALYSIS
# ──────────────────────────────────────────────

def get_terrain_analysis(aoi: ee.Geometry) -> Dict:
    """Get elevation and slope data from SRTM."""
    try:
        srtm = ee.Image("USGS/SRTMGL1_003").clip(aoi)
        slope = ee.Terrain.slope(srtm)

        elev_stats = srtm.reduceRegion(
            reducer=ee.Reducer.mean().combine(
                ee.Reducer.min(), sharedInputs=True
            ).combine(
                ee.Reducer.max(), sharedInputs=True
            ),
            geometry=aoi,
            scale=30,
            maxPixels=1e10,
            bestEffort=True
        ).getInfo()

        slope_stats = slope.reduceRegion(
            reducer=ee.Reducer.mean().combine(
                ee.Reducer.max(), sharedInputs=True
            ),
            geometry=aoi,
            scale=30,
            maxPixels=1e10,
            bestEffort=True
        ).getInfo()

        return {
            "mean_elevation_m": round(elev_stats.get("elevation_mean", 0), 1),
            "min_elevation_m": round(elev_stats.get("elevation_min", 0), 1),
            "max_elevation_m": round(elev_stats.get("elevation_max", 0), 1),
            "mean_slope_deg": round(slope_stats.get("slope_mean", 0), 1),
            "max_slope_deg": round(slope_stats.get("slope_max", 0), 1),
        }
    except Exception as e:
        print(f"Terrain analysis error: {e}")
        return {
            "mean_elevation_m": 0, "min_elevation_m": 0,
            "max_elevation_m": 0, "mean_slope_deg": 0, "max_slope_deg": 0
        }


# ──────────────────────────────────────────────
# CLIMATE / RAINFALL
# ──────────────────────────────────────────────

def get_climate_data(aoi: ee.Geometry) -> Dict:
    """Get mean annual rainfall and temperature from WorldClim/CHIRPS."""
    try:
        # Annual precipitation from CHIRPS
        chirps = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY") \
            .filterBounds(aoi) \
            .filterDate("2020-01-01", "2020-12-31")

        annual_precip = chirps.sum().clip(aoi)
        precip_stats = annual_precip.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=aoi,
            scale=5000,
            maxPixels=1e10,
            bestEffort=True
        ).getInfo()

        # Temperature from ERA5 monthly
        era5 = ee.ImageCollection("ECMWF/ERA5_LAND/MONTHLY_AGGR") \
            .filterBounds(aoi) \
            .filterDate("2020-01-01", "2020-12-31") \
            .select("temperature_2m")

        mean_temp = era5.mean().clip(aoi)
        temp_stats = mean_temp.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=aoi,
            scale=11000,
            maxPixels=1e10,
            bestEffort=True
        ).getInfo()

        temp_c = (temp_stats.get("temperature_2m", 273.15) - 273.15)

        return {
            "annual_rainfall_mm": round(precip_stats.get("precipitation", 0), 0),
            "mean_temperature_c": round(temp_c, 1),
            "climate_zone": classify_climate_zone(
                precip_stats.get("precipitation", 0), temp_c
            ),
        }
    except Exception as e:
        print(f"Climate analysis error: {e}")
        return {"annual_rainfall_mm": 0, "mean_temperature_c": 0, "climate_zone": "Unknown"}


def classify_climate_zone(rainfall_mm: float, temp_c: float) -> str:
    """Simple climate zone classification for carbon methodology."""
    if temp_c > 18:
        if rainfall_mm > 2000:
            return "Tropical Wet"
        elif rainfall_mm > 1000:
            return "Tropical Moist"
        else:
            return "Tropical Dry"
    elif temp_c > 10:
        if rainfall_mm > 1000:
            return "Subtropical Moist"
        else:
            return "Subtropical Dry"
    else:
        if rainfall_mm > 600:
            return "Temperate Moist"
        else:
            return "Temperate Dry"


# ──────────────────────────────────────────────
# PROTECTED AREA CHECK
# ──────────────────────────────────────────────

def check_protected_areas(aoi: ee.Geometry) -> Dict:
    """Check if AOI overlaps with WDPA protected areas."""
    try:
        wdpa = ee.FeatureCollection("WCMC/WDPA/current/polygons")
        overlapping = wdpa.filterBounds(aoi)
        count = overlapping.size().getInfo()

        if count > 0:
            names = overlapping.aggregate_array("NAME").getInfo()[:5]
            categories = overlapping.aggregate_array("IUCN_CAT").getInfo()[:5]
            return {
                "overlaps_protected_area": True,
                "protected_area_count": count,
                "names": names,
                "iucn_categories": categories,
            }
        return {"overlaps_protected_area": False, "protected_area_count": 0}
    except Exception as e:
        print(f"Protected area check error: {e}")
        return {"overlaps_protected_area": False, "protected_area_count": 0}


# ──────────────────────────────────────────────
# MASTER ANALYSIS FUNCTION
# ──────────────────────────────────────────────

def run_full_analysis(geojson_data: dict, analysis_year: int = None) -> Dict:
    """
    Run the complete pre-feasibility spatial analysis.
    Returns all data needed for eligibility determination.
    """
    if analysis_year is None:
        analysis_year = datetime.now().year

    start_year = analysis_year - 10
    aoi = geometry_from_geojson(geojson_data)

    # Get centroid for location info
    centroid = aoi.centroid().coordinates().getInfo()

    results = {
        "metadata": {
            "analysis_date": datetime.now().isoformat(),
            "analysis_period": f"{start_year}-{analysis_year}",
            "centroid_lon": round(centroid[0], 6),
            "centroid_lat": round(centroid[1], 6),
        },
        "lulc_timeseries": {},
        "hansen_forest": {},
        "mangrove": {},
        "ndvi_timeseries": {},
        "terrain": {},
        "climate": {},
        "protected_areas": {},
        "soil_carbon": {},
    }

    # Run analyses
    print("Running LULC time series analysis...")
    results["lulc_timeseries"] = get_esri_lulc_timeseries(aoi, start_year, analysis_year)

    print("Running Hansen forest change analysis...")
    results["hansen_forest"] = analyze_hansen_forest(aoi)

    print("Running mangrove analysis...")
    results["mangrove"] = analyze_mangroves(aoi)

    print("Running NDVI time series...")
    results["ndvi_timeseries"] = get_ndvi_timeseries(aoi, start_year, analysis_year)

    print("Running terrain analysis...")
    results["terrain"] = get_terrain_analysis(aoi)

    print("Running climate analysis...")
    results["climate"] = get_climate_data(aoi)

    print("Checking protected areas...")
    results["protected_areas"] = check_protected_areas(aoi)

    print("Getting soil carbon data...")
    results["soil_carbon"] = get_soil_carbon(aoi)

    return results
