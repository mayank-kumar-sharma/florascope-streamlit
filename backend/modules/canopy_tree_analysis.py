from __future__ import annotations

"""
Canopy Cover & Tree Analysis Module (GEE-based)

Features:
  1. Canopy cover percentage over years (Hansen treecover + yearly updates)
  2. Tree density estimation via NDVI segmentation
  3. Canopy height model from GEDI/ALOS
  4. Above-ground biomass estimation
  5. Tree count estimation from high-res imagery
  6. Crown area detection
"""

import json
from datetime import datetime
from typing import Dict, List, Optional

try:
    import ee
except ImportError:
    ee = None  # GEE not available — demo functions still work


# ──────────────────────────────────────────────
# CANOPY COVER OVER YEARS
# ──────────────────────────────────────────────

def analyze_canopy_cover_timeseries(aoi: ee.Geometry, start_year: int, end_year: int) -> Dict:
    """
    Calculate canopy cover percentage for each year using:
    - Hansen Global Forest Change (baseline 2000 + annual loss/gain)
    - MODIS VCF (Vegetation Continuous Fields) for annual tree cover %
    - Sentinel-2 NDVI-based canopy estimation for recent years
    """
    results = {
        "hansen_baseline": {},
        "modis_vcf_timeseries": {},
        "sentinel_ndvi_canopy": {},
        "summary": {},
    }

    pixel_area = ee.Image.pixelArea()

    # ── Hansen Baseline + Adjusted ──
    hansen = ee.Image("UMD/hansen/global_forest_change_2023_v1_11")
    tree_cover_2000 = hansen.select("treecover2000")
    loss_year = hansen.select("lossyear")
    gain = hansen.select("gain")

    # Baseline stats
    tc_stats = tree_cover_2000.reduceRegion(
        reducer=ee.Reducer.mean().combine(
            ee.Reducer.percentile([10, 25, 50, 75, 90]), sharedInputs=True
        ),
        geometry=aoi, scale=30, maxPixels=1e10, bestEffort=True
    ).getInfo()

    results["hansen_baseline"] = {
        "mean_canopy_pct": round(tc_stats.get("treecover2000_mean", 0), 1),
        "median_canopy_pct": round(tc_stats.get("treecover2000_p50", 0), 1),
        "p10": round(tc_stats.get("treecover2000_p10", 0), 1),
        "p90": round(tc_stats.get("treecover2000_p90", 0), 1),
    }

    # Reconstruct canopy cover for each year
    # Year Y canopy = 2000 canopy - losses(2001..Y) + gain (binary, so approximated)
    canopy_by_year = {}
    total_area = pixel_area.reduceRegion(
        reducer=ee.Reducer.sum(), geometry=aoi,
        scale=30, maxPixels=1e10, bestEffort=True
    ).get("area").getInfo() or 1

    for year in range(max(2001, start_year), min(end_year + 1, 2024)):
        yr_code = year - 2000
        # Cumulative loss up to this year
        cum_loss = loss_year.gt(0).And(loss_year.lte(yr_code))
        # Adjusted canopy: original minus cumulative loss pixels
        adjusted = tree_cover_2000.updateMask(cum_loss.Not())

        mean_canopy = adjusted.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=aoi, scale=30, maxPixels=1e10, bestEffort=True
        ).get("treecover2000")

        # Forest area (>30% canopy)
        forest_mask = adjusted.gte(30)
        forest_area = pixel_area.updateMask(forest_mask).reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=aoi, scale=30, maxPixels=1e10, bestEffort=True
        ).get("area")

        canopy_by_year[year] = {
            "mean_canopy_pct": round((mean_canopy.getInfo() or 0), 1),
            "forest_area_ha": round((forest_area.getInfo() or 0) / 10000, 2),
            "forest_cover_pct": round(((forest_area.getInfo() or 0) / total_area) * 100, 1),
        }

    results["hansen_adjusted_timeseries"] = canopy_by_year

    # ── MODIS VCF (500m, annual tree cover %) ──
    try:
        vcf_timeseries = {}
        for year in range(max(2000, start_year), min(end_year + 1, 2024)):
            vcf = ee.ImageCollection("MODIS/006/MOD44B") \
                .filterDate(f"{year}-01-01", f"{year}-12-31") \
                .first()

            if vcf:
                tc = vcf.select("Percent_Tree_Cover").clip(aoi)
                mean_tc = tc.reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=aoi, scale=500, maxPixels=1e10, bestEffort=True
                ).get("Percent_Tree_Cover")

                vcf_timeseries[year] = round(mean_tc.getInfo() or 0, 1)

        results["modis_vcf_timeseries"] = vcf_timeseries
    except Exception as e:
        print(f"MODIS VCF error: {e}")

    # ── Sentinel-2 NDVI-based Canopy (10m, 2017+) ──
    try:
        ndvi_canopy = {}
        for year in range(max(2017, start_year), end_year + 1):
            s2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED") \
                .filterBounds(aoi) \
                .filterDate(f"{year}-06-01", f"{year}-09-30") \
                .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 15))

            if s2.size().getInfo() == 0:
                # Try full year
                s2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED") \
                    .filterBounds(aoi) \
                    .filterDate(f"{year}-01-01", f"{year}-12-31") \
                    .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))

            ndvi = s2.map(lambda img: img.normalizedDifference(["B8", "B4"]).rename("NDVI")).median().clip(aoi)

            # Canopy proxy: NDVI > 0.4 generally indicates tree canopy
            canopy_mask = ndvi.gt(0.4)
            canopy_area = pixel_area.updateMask(canopy_mask).reduceRegion(
                reducer=ee.Reducer.sum(),
                geometry=aoi, scale=10, maxPixels=1e10, bestEffort=True
            ).get("area")

            # Dense canopy: NDVI > 0.6
            dense_mask = ndvi.gt(0.6)
            dense_area = pixel_area.updateMask(dense_mask).reduceRegion(
                reducer=ee.Reducer.sum(),
                geometry=aoi, scale=10, maxPixels=1e10, bestEffort=True
            ).get("area")

            mean_ndvi = ndvi.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=aoi, scale=10, maxPixels=1e10, bestEffort=True
            ).get("NDVI")

            ndvi_canopy[year] = {
                "canopy_area_ha": round((canopy_area.getInfo() or 0) / 10000, 2),
                "canopy_cover_pct": round(((canopy_area.getInfo() or 0) / total_area) * 100, 1),
                "dense_canopy_ha": round((dense_area.getInfo() or 0) / 10000, 2),
                "mean_ndvi": round(mean_ndvi.getInfo() or 0, 3),
            }

        results["sentinel_ndvi_canopy"] = ndvi_canopy
    except Exception as e:
        print(f"Sentinel canopy error: {e}")

    # ── Summary ──
    years = sorted(canopy_by_year.keys())
    if len(years) >= 2:
        first_yr = canopy_by_year[years[0]]
        last_yr = canopy_by_year[years[-1]]
        results["summary"] = {
            "canopy_change_pct": round(
                last_yr["mean_canopy_pct"] - first_yr["mean_canopy_pct"], 1
            ),
            "forest_area_change_ha": round(
                last_yr["forest_area_ha"] - first_yr["forest_area_ha"], 2
            ),
            "trend": "increasing" if last_yr["mean_canopy_pct"] > first_yr["mean_canopy_pct"]
                     else "decreasing" if last_yr["mean_canopy_pct"] < first_yr["mean_canopy_pct"]
                     else "stable",
            "first_year": years[0],
            "last_year": years[-1],
        }

    return results


# ──────────────────────────────────────────────
# TREE COUNT ESTIMATION
# ──────────────────────────────────────────────

def estimate_tree_count(aoi: ee.Geometry, year: int = None) -> Dict:
    """
    Estimate tree count using multiple approaches:
    1. NDVI peak detection (local maxima = individual tree crowns)
    2. Canopy area / average crown size
    3. Tree density from literature by vegetation type
    """
    if year is None:
        year = datetime.now().year

    results = {
        "ndvi_method": {},
        "crown_area_method": {},
        "combined_estimate": {},
    }

    pixel_area = ee.Image.pixelArea()
    total_area_m2 = pixel_area.reduceRegion(
        reducer=ee.Reducer.sum(), geometry=aoi,
        scale=10, maxPixels=1e10, bestEffort=True
    ).get("area").getInfo() or 0
    total_area_ha = total_area_m2 / 10000

    # ── Method 1: NDVI Local Maxima ──
    try:
        # Use Sentinel-2 for highest resolution
        s2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED") \
            .filterBounds(aoi) \
            .filterDate(f"{year}-01-01", f"{year}-12-31") \
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 15)) \
            .sort("CLOUDY_PIXEL_PERCENTAGE")

        ndvi = s2.map(
            lambda img: img.normalizedDifference(["B8", "B4"]).rename("NDVI")
        ).median().clip(aoi)

        # Tree canopy pixels (NDVI > 0.4)
        tree_mask = ndvi.gt(0.4)
        tree_area_m2 = pixel_area.updateMask(tree_mask).reduceRegion(
            reducer=ee.Reducer.sum(), geometry=aoi,
            scale=10, maxPixels=1e10, bestEffort=True
        ).get("area").getInfo() or 0

        # Count tree pixels (each 10m pixel = 100 sq m)
        tree_pixel_count = tree_mask.reduceRegion(
            reducer=ee.Reducer.sum(), geometry=aoi,
            scale=10, maxPixels=1e10, bestEffort=True
        ).get("NDVI").getInfo() or 0

        # Local maxima detection (approximation)
        # Apply Gaussian smoothing then find peaks
        smooth_ndvi = ndvi.focal_mean(radius=15, units="meters")
        peaks = ndvi.subtract(smooth_ndvi).gt(0.05).And(tree_mask)
        peak_count = peaks.reduceRegion(
            reducer=ee.Reducer.sum(), geometry=aoi,
            scale=10, maxPixels=1e10, bestEffort=True
        ).get("NDVI").getInfo() or 0

        results["ndvi_method"] = {
            "tree_pixels_10m": int(tree_pixel_count),
            "canopy_area_ha": round(tree_area_m2 / 10000, 2),
            "local_maxima_count": int(peak_count),
            "estimated_trees": int(peak_count),
            "note": "Based on NDVI local maxima from 10m Sentinel-2 imagery"
        }
    except Exception as e:
        results["ndvi_method"] = {"error": str(e)}

    # ── Method 2: Crown Area Division ──
    try:
        canopy_ha = tree_area_m2 / 10000 if tree_area_m2 else 0
        canopy_m2 = canopy_ha * 10000

        # Average crown areas by biome (sq meters)
        crown_estimates = {
            "small_trees_3m_crown": {"crown_m2": 7, "count": int(canopy_m2 / 7) if canopy_m2 else 0},
            "medium_trees_5m_crown": {"crown_m2": 20, "count": int(canopy_m2 / 20) if canopy_m2 else 0},
            "large_trees_8m_crown": {"crown_m2": 50, "count": int(canopy_m2 / 50) if canopy_m2 else 0},
            "mixed_average_6m_crown": {"crown_m2": 28, "count": int(canopy_m2 / 28) if canopy_m2 else 0},
        }

        results["crown_area_method"] = {
            "canopy_area_m2": round(canopy_m2, 0),
            "estimates_by_crown_size": crown_estimates,
            "recommended_estimate": crown_estimates["mixed_average_6m_crown"]["count"],
            "note": "Divides total canopy area by average crown size"
        }
    except Exception as e:
        results["crown_area_method"] = {"error": str(e)}

    # ── Combined Estimate ──
    ndvi_count = results.get("ndvi_method", {}).get("estimated_trees", 0)
    crown_count = results.get("crown_area_method", {}).get("recommended_estimate", 0)

    if ndvi_count > 0 and crown_count > 0:
        avg_estimate = int((ndvi_count + crown_count) / 2)
    else:
        avg_estimate = max(ndvi_count, crown_count)

    density = avg_estimate / max(total_area_ha, 0.001) if avg_estimate > 0 else 0

    results["combined_estimate"] = {
        "estimated_tree_count": avg_estimate,
        "low_estimate": min(ndvi_count, crown_count) if ndvi_count and crown_count else avg_estimate,
        "high_estimate": max(ndvi_count, crown_count) if ndvi_count and crown_count else avg_estimate,
        "trees_per_hectare": round(density, 0),
        "total_area_ha": round(total_area_ha, 2),
        "canopy_cover_pct": round((tree_area_m2 / max(total_area_m2, 1)) * 100, 1) if tree_area_m2 else 0,
    }

    return results


# ──────────────────────────────────────────────
# CANOPY HEIGHT MODEL
# ──────────────────────────────────────────────

def analyze_canopy_height(aoi: ee.Geometry) -> Dict:
    """
    Analyze canopy height using:
    - Global Canopy Height (ETH/Meta 2020)
    - GEDI L2A (if available)
    - ALOS PALSAR biomass proxy
    """
    results = {}

    # ── ETH Global Canopy Height (10m, 2020) ──
    try:
        canopy_height = ee.Image("users/nlang/ETH_GlobalCanopyHeight_2020_10m_v1") \
            .clip(aoi)

        stats = canopy_height.reduceRegion(
            reducer=ee.Reducer.mean().combine(
                ee.Reducer.max(), sharedInputs=True
            ).combine(
                ee.Reducer.percentile([25, 50, 75, 90]), sharedInputs=True
            ),
            geometry=aoi, scale=10, maxPixels=1e10, bestEffort=True
        ).getInfo()

        # Get first band name dynamically
        band = list(stats.keys())[0].split("_")[0] if stats else "b1"
        results["eth_canopy_height"] = {
            "mean_height_m": round(stats.get(f"{band}_mean", 0) or 0, 1),
            "max_height_m": round(stats.get(f"{band}_max", 0) or 0, 1),
            "median_height_m": round(stats.get(f"{band}_p50", 0) or 0, 1),
            "p75_height_m": round(stats.get(f"{band}_p75", 0) or 0, 1),
            "p90_height_m": round(stats.get(f"{band}_p90", 0) or 0, 1),
            "source": "ETH Global Canopy Height 2020 (10m)",
        }
    except Exception as e:
        print(f"Canopy height error: {e}")
        # Fallback: try Meta/WRI global tree height
        try:
            canopy_height = ee.ImageCollection(
                "projects/meta-forest-monitoring-okw37/assets/v1"
            ).mosaic().clip(aoi)
            stats = canopy_height.reduceRegion(
                reducer=ee.Reducer.mean().combine(ee.Reducer.max(), sharedInputs=True),
                geometry=aoi, scale=10, maxPixels=1e10, bestEffort=True
            ).getInfo()
            results["canopy_height"] = {
                "mean_height_m": round(list(stats.values())[0] or 0, 1) if stats else 0,
                "source": "Meta/WRI Canopy Height"
            }
        except:
            results["canopy_height"] = {"error": "Canopy height data unavailable"}

    return results


# ──────────────────────────────────────────────
# ABOVE-GROUND BIOMASS ESTIMATION
# ──────────────────────────────────────────────

def estimate_biomass(aoi: ee.Geometry) -> Dict:
    """
    Estimate above-ground biomass (AGB) using:
    - ESA CCI Biomass (2017-2020, 100m)
    - GlobBiomass (2010, 100m)
    - NDVI-based allometric estimation
    """
    results = {}

    # ── ESA CCI Biomass ──
    try:
        cci_biomass = ee.ImageCollection("ESA/CCI/FireCCI/5_1").first()  # placeholder
        # Try NASA GEDI L4B
        gedi = ee.Image("LARSE/GEDI/GEDI04_B_002").clip(aoi)
        stats = gedi.select("MU").reduceRegion(
            reducer=ee.Reducer.mean().combine(ee.Reducer.sum(), sharedInputs=True),
            geometry=aoi, scale=1000, maxPixels=1e10, bestEffort=True
        ).getInfo()

        results["gedi_biomass"] = {
            "mean_agb_mg_per_ha": round(stats.get("MU_mean", 0) or 0, 1),
            "total_agb_mg": round(stats.get("MU_sum", 0) or 0, 0),
            "source": "GEDI L4B"
        }
    except Exception as e:
        results["gedi_biomass"] = {"error": str(e), "note": "GEDI data may not cover this area"}

    # ── NDVI-based Allometric Estimation ──
    try:
        year = datetime.now().year
        s2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED") \
            .filterBounds(aoi) \
            .filterDate(f"{year-1}-01-01", f"{year-1}-12-31") \
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))

        ndvi = s2.map(
            lambda img: img.normalizedDifference(["B8", "B4"]).rename("NDVI")
        ).median().clip(aoi)

        # Simple allometric: AGB ≈ 200 * (NDVI)^2 for tropical forests (rough proxy)
        agb_proxy = ndvi.pow(2).multiply(200).rename("AGB_proxy")

        agb_stats = agb_proxy.reduceRegion(
            reducer=ee.Reducer.mean().combine(ee.Reducer.sum(), sharedInputs=True),
            geometry=aoi, scale=10, maxPixels=1e10, bestEffort=True
        ).getInfo()

        pixel_area = ee.Image.pixelArea()
        total_area = pixel_area.reduceRegion(
            reducer=ee.Reducer.sum(), geometry=aoi,
            scale=10, maxPixels=1e10, bestEffort=True
        ).get("area").getInfo() or 0

        results["ndvi_biomass"] = {
            "mean_agb_proxy_mg_per_ha": round(agb_stats.get("AGB_proxy_mean", 0) or 0, 1),
            "note": "NDVI-based proxy (allometric). For accurate AGB, use field measurements.",
            "source": "Sentinel-2 NDVI allometric proxy"
        }
    except Exception as e:
        results["ndvi_biomass"] = {"error": str(e)}

    # ── Carbon Stock Estimation ──
    # AGB to carbon: multiply by 0.47 (IPCC default)
    # Carbon to CO2e: multiply by 3.67
    mean_agb = results.get("gedi_biomass", {}).get("mean_agb_mg_per_ha",
               results.get("ndvi_biomass", {}).get("mean_agb_proxy_mg_per_ha", 0))

    if mean_agb > 0:
        carbon_per_ha = mean_agb * 0.47
        co2e_per_ha = carbon_per_ha * 3.67
        pixel_area_val = ee.Image.pixelArea().reduceRegion(
            reducer=ee.Reducer.sum(), geometry=aoi,
            scale=30, maxPixels=1e10, bestEffort=True
        ).get("area").getInfo() or 0
        area_ha = pixel_area_val / 10000

        results["carbon_stock"] = {
            "mean_carbon_per_ha_tC": round(carbon_per_ha, 1),
            "mean_co2e_per_ha": round(co2e_per_ha, 1),
            "total_carbon_tC": round(carbon_per_ha * area_ha, 0),
            "total_co2e": round(co2e_per_ha * area_ha, 0),
            "area_ha": round(area_ha, 2),
            "note": "Based on AGB × 0.47 (IPCC carbon fraction) × 3.67 (CO2 conversion)"
        }

    return results


# ──────────────────────────────────────────────
# DEMO/MOCK FUNCTIONS
# ──────────────────────────────────────────────

def demo_canopy_analysis(total_area_ha: float) -> Dict:
    """Generate realistic mock canopy analysis for demo mode."""
    import random
    random.seed(42)

    current_year = datetime.now().year
    start_year = current_year - 10
    base_canopy = random.uniform(10, 35)

    hansen_ts = {}
    for year in range(start_year, current_year + 1):
        drift = random.uniform(-0.5, 0.8) * (year - start_year) / 10
        canopy = max(0, min(100, base_canopy + drift + random.gauss(0, 1)))
        forest_pct = max(0, canopy - 5 + random.gauss(0, 2))
        hansen_ts[year] = {
            "mean_canopy_pct": round(canopy, 1),
            "forest_area_ha": round(total_area_ha * forest_pct / 100, 2),
            "forest_cover_pct": round(forest_pct, 1),
        }

    sentinel_canopy = {}
    for year in range(max(2017, start_year), current_year + 1):
        canopy_pct = hansen_ts[year]["mean_canopy_pct"] + random.gauss(0, 2)
        sentinel_canopy[year] = {
            "canopy_area_ha": round(total_area_ha * canopy_pct / 100, 2),
            "canopy_cover_pct": round(canopy_pct, 1),
            "dense_canopy_ha": round(total_area_ha * canopy_pct * 0.3 / 100, 2),
            "mean_ndvi": round(0.2 + canopy_pct * 0.008 + random.gauss(0, 0.02), 3),
        }

    tree_est = int(total_area_ha * random.uniform(80, 350))

    return {
        "canopy_cover": {
            "hansen_baseline": {
                "mean_canopy_pct": round(base_canopy, 1),
                "median_canopy_pct": round(base_canopy - 2, 1),
            },
            "hansen_adjusted_timeseries": hansen_ts,
            "sentinel_ndvi_canopy": sentinel_canopy,
            "summary": {
                "canopy_change_pct": round(
                    hansen_ts[current_year - 1]["mean_canopy_pct"] -
                    hansen_ts[start_year]["mean_canopy_pct"], 1
                ),
                "trend": "increasing" if hansen_ts[current_year - 1]["mean_canopy_pct"] >
                         hansen_ts[start_year]["mean_canopy_pct"] else "decreasing",
            }
        },
        "tree_count": {
            "combined_estimate": {
                "estimated_tree_count": tree_est,
                "trees_per_hectare": round(tree_est / max(total_area_ha, 1), 0),
                "canopy_cover_pct": round(base_canopy, 1),
                "total_area_ha": round(total_area_ha, 2),
            },
        },
        "canopy_height": {
            "eth_canopy_height": {
                "mean_height_m": round(random.uniform(3, 18), 1),
                "max_height_m": round(random.uniform(15, 30), 1),
                "median_height_m": round(random.uniform(4, 15), 1),
            }
        },
        "biomass": {
            "carbon_stock": {
                "mean_carbon_per_ha_tC": round(random.uniform(20, 120), 1),
                "mean_co2e_per_ha": round(random.uniform(75, 440), 1),
                "total_carbon_tC": round(random.uniform(20, 120) * total_area_ha, 0),
                "total_co2e": round(random.uniform(75, 440) * total_area_ha, 0),
            }
        }
    }
