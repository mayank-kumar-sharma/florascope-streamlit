"""
Gold Standard Eligibility Engine
Implements decision logic per:
  - GS4GG A/R LUF Activity Requirements (v1.2.1, Document 203)
  - GS A/R Methodology (Simplified baseline and monitoring for small-scale CDM A/R)
  - GS Blue Carbon / Mangrove Methodology (BCFW, Document 443)

Determines project type suitability: ARR, IFM, REDD+, Blue Carbon (Mangrove)
"""

from typing import Dict, List, Tuple
from dataclasses import dataclass, field
from enum import Enum


class ProjectType(str, Enum):
    ARR = "ARR"                      # Afforestation / Reforestation / Revegetation
    IFM = "IFM"                      # Improved Forest Management
    REDD = "REDD+"                   # Reducing Emissions from Deforestation & Degradation
    BLUE_CARBON = "Blue Carbon"      # Mangrove / Tidal Wetland Restoration
    AGF = "Agroforestry"             # Agroforestry on agricultural land
    NOT_ELIGIBLE = "Not Eligible"


class EligibilityStatus(str, Enum):
    HIGHLY_SUITABLE = "Highly Suitable"
    SUITABLE = "Suitable"
    POSSIBLY_SUITABLE = "Possibly Suitable"
    NOT_SUITABLE = "Not Suitable"
    NEEDS_INVESTIGATION = "Needs Further Investigation"


@dataclass
class EligibilityResult:
    project_type: ProjectType
    status: EligibilityStatus
    confidence: float  # 0-100
    reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    gs_requirements_met: Dict[str, bool] = field(default_factory=dict)
    recommended_methodology: str = ""
    estimated_potential: str = ""


@dataclass
class LandProfile:
    """Summarized land characteristics derived from GEE analysis."""
    total_area_ha: float = 0
    # Current land use
    current_forest_pct: float = 0
    current_shrub_grass_pct: float = 0
    current_crop_pct: float = 0
    current_bare_pct: float = 0
    current_wetland_pct: float = 0
    current_water_pct: float = 0
    current_built_pct: float = 0
    current_mangrove_pct: float = 0
    # Historical
    was_forest_10yr_ago: bool = False
    forest_pct_10yr_ago: float = 0
    has_deforestation: bool = False
    deforestation_ha: float = 0
    deforestation_rate_pct_yr: float = 0
    years_non_forest: int = 0
    # Hansen
    hansen_forest_cover_2000_pct: float = 0
    hansen_loss_ha: float = 0
    hansen_gain_ha: float = 0
    # Mangrove
    mangrove_present: bool = False
    mangrove_area_ha: float = 0
    mangrove_loss_ha: float = 0
    historical_mangrove: bool = False
    tidal_wetland: bool = False
    # Climate & terrain
    climate_zone: str = ""
    annual_rainfall_mm: float = 0
    mean_elevation_m: float = 0
    mean_slope_deg: float = 0
    mean_ndvi: float = 0
    # Protected areas
    in_protected_area: bool = False
    # Soil
    soil_carbon_g_per_kg: float = 0


def build_land_profile(analysis_results: Dict) -> LandProfile:
    """Build a LandProfile from raw GEE analysis results."""
    profile = LandProfile()

    # ── Total area ──
    hansen = analysis_results.get("hansen_forest", {})
    profile.total_area_ha = hansen.get("total_area_ha", 0)

    # ── Current LULC (latest year available) ──
    lulc = analysis_results.get("lulc_timeseries", {})
    years = sorted(lulc.keys(), reverse=True)

    if years:
        latest = lulc[years[0]]
        total_lulc_ha = sum(latest.values()) or 1

        profile.current_forest_pct = (latest.get("Trees", 0) / total_lulc_ha) * 100
        profile.current_mangrove_pct = (latest.get("Mangroves", 0) / total_lulc_ha) * 100
        profile.current_shrub_grass_pct = (
            (latest.get("Rangeland", 0) + latest.get("Grass", 0) +
             latest.get("Shrub & Scrub", 0)) / total_lulc_ha
        ) * 100
        profile.current_crop_pct = (latest.get("Crops", 0) / total_lulc_ha) * 100
        profile.current_bare_pct = (latest.get("Bare Ground", 0) / total_lulc_ha) * 100
        profile.current_wetland_pct = (
            (latest.get("Flooded Vegetation", 0)) / total_lulc_ha
        ) * 100
        profile.current_water_pct = (latest.get("Water", 0) / total_lulc_ha) * 100
        profile.current_built_pct = (latest.get("Built Area", 0) / total_lulc_ha) * 100

    # ── Historical LULC (10 years ago) ──
    if len(years) >= 2:
        earliest_key = years[-1]
        earliest = lulc[earliest_key]
        total_earliest_ha = sum(earliest.values()) or 1

        profile.forest_pct_10yr_ago = (
            (earliest.get("Trees", 0) + earliest.get("Mangroves", 0)) / total_earliest_ha
        ) * 100
        profile.was_forest_10yr_ago = profile.forest_pct_10yr_ago > 30

        # Detect deforestation: more forest before than now
        if profile.forest_pct_10yr_ago > profile.current_forest_pct + 10:
            profile.has_deforestation = True

    # Count consecutive years of non-forest
    non_forest_count = 0
    for yr_key in sorted(years, reverse=True):
        yr_data = lulc[yr_key]
        yr_total = sum(yr_data.values()) or 1
        forest_pct = ((yr_data.get("Trees", 0) + yr_data.get("Mangroves", 0)) / yr_total) * 100
        if forest_pct < 20:
            non_forest_count += 1
        else:
            break
    profile.years_non_forest = non_forest_count

    # ── Hansen data ──
    profile.hansen_forest_cover_2000_pct = hansen.get("forest_cover_pct_2000", 0)
    profile.hansen_loss_ha = hansen.get("forest_loss_total_ha", 0)
    profile.hansen_gain_ha = hansen.get("forest_gain_ha", 0)
    profile.deforestation_ha = profile.hansen_loss_ha

    if profile.total_area_ha > 0 and profile.hansen_loss_ha > 0:
        # Average annual deforestation rate over the analysis period
        profile.deforestation_rate_pct_yr = (
            (profile.hansen_loss_ha / max(profile.total_area_ha, 1)) / 23 * 100
        )

    # ── Mangrove ──
    mangrove = analysis_results.get("mangrove", {})
    profile.mangrove_present = mangrove.get("mangrove_present", False)
    profile.mangrove_area_ha = mangrove.get("mangrove_area_ha", 0)
    profile.mangrove_loss_ha = mangrove.get("mangrove_loss_ha", 0)
    profile.historical_mangrove = mangrove.get("historical_mangrove", False)
    profile.tidal_wetland = mangrove.get("tidal_wetland", False)

    # ── Climate & Terrain ──
    climate = analysis_results.get("climate", {})
    profile.climate_zone = climate.get("climate_zone", "")
    profile.annual_rainfall_mm = climate.get("annual_rainfall_mm", 0)

    terrain = analysis_results.get("terrain", {})
    profile.mean_elevation_m = terrain.get("mean_elevation_m", 0)
    profile.mean_slope_deg = terrain.get("mean_slope_deg", 0)

    # Mean NDVI (latest available)
    ndvi = analysis_results.get("ndvi_timeseries", {})
    ndvi_values = [v for v in ndvi.values() if v is not None]
    profile.mean_ndvi = sum(ndvi_values) / len(ndvi_values) if ndvi_values else 0

    # Protected areas
    pa = analysis_results.get("protected_areas", {})
    profile.in_protected_area = pa.get("overlaps_protected_area", False)

    # Soil carbon
    soc = analysis_results.get("soil_carbon", {})
    profile.soil_carbon_g_per_kg = soc.get("mean_soc_g_per_kg", 0)

    return profile


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ARR ELIGIBILITY (Gold Standard A/R)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def check_arr_eligibility(profile: LandProfile) -> EligibilityResult:
    """
    Check eligibility for Afforestation/Reforestation/Revegetation.

    Per GS LUF Activity Requirements (Doc 203, v1.2.1):
    - Land must NOT have been forest for at least 10 years prior to project start
    - Land must not have been cleared of native ecosystems to create A/R conditions
    - Must demonstrate land was non-forest (grassland, cropland, degraded, bare)
    - Must not be wetland (flooded vegetation, permanent water)
    - Small-scale: <8000 ha for A/R CDM methodology

    Per GS A/R Methodology:
    - Baseline: non-forest land
    - Project area dominated by non-tree vegetation or bare
    - Evidence of degradation preferred (low NDVI, bare ground)
    """
    result = EligibilityResult(
        project_type=ProjectType.ARR,
        status=EligibilityStatus.NOT_SUITABLE,
        confidence=0,
        recommended_methodology="GS Simplified Baseline & Monitoring for Small-Scale CDM A/R"
    )

    score = 0
    max_score = 100

    # ── Requirement 1: Land is currently non-forest (CRITICAL) ──
    if profile.current_forest_pct < 20:
        result.gs_requirements_met["Current land is non-forest (<20% tree cover)"] = True
        result.reasons.append(
            f"Current forest cover is {profile.current_forest_pct:.1f}% — classified as non-forest land"
        )
        score += 30
    elif profile.current_forest_pct < 40:
        result.gs_requirements_met["Current land is non-forest (<20% tree cover)"] = False
        result.reasons.append(
            f"Current forest cover is {profile.current_forest_pct:.1f}% — partially forested, may not qualify"
        )
        result.warnings.append("Tree cover between 20-40% — requires field verification of forest definition")
        score += 10
    else:
        result.gs_requirements_met["Current land is non-forest (<20% tree cover)"] = False
        result.reasons.append(
            f"Current forest cover is {profile.current_forest_pct:.1f}% — land is already forested"
        )
        score -= 20

    # ── Requirement 2: Land was non-forest for at least 10 years (CRITICAL) ──
    if profile.years_non_forest >= 10:
        result.gs_requirements_met["Non-forest for ≥10 years"] = True
        result.reasons.append(
            f"Land has been non-forest for {profile.years_non_forest} consecutive years"
        )
        score += 25
    elif not profile.was_forest_10yr_ago and profile.hansen_forest_cover_2000_pct < 30:
        result.gs_requirements_met["Non-forest for ≥10 years"] = True
        result.reasons.append(
            "Hansen data confirms low forest cover in 2000, supporting long-term non-forest status"
        )
        score += 20
    else:
        result.gs_requirements_met["Non-forest for ≥10 years"] = False
        result.warnings.append(
            "Cannot confirm 10-year non-forest history — may need historical imagery review"
        )
        score += 5

    # ── Requirement 3: No clearing of native ecosystem (CRITICAL) ──
    if not profile.has_deforestation and profile.hansen_loss_ha < (profile.total_area_ha * 0.05):
        result.gs_requirements_met["No evidence of deliberate clearing"] = True
        result.reasons.append("No significant deforestation detected in the analysis period")
        score += 20
    else:
        result.gs_requirements_met["No evidence of deliberate clearing"] = False
        result.warnings.append(
            f"Deforestation of {profile.deforestation_ha:.1f} ha detected — "
            "must prove land was not intentionally cleared for A/R project"
        )
        score -= 10

    # ── Requirement 4: Not wetland/permanent water ──
    if profile.current_wetland_pct < 10 and profile.current_water_pct < 10:
        result.gs_requirements_met["Land is not wetland/water"] = True
        score += 10
    else:
        result.gs_requirements_met["Land is not wetland/water"] = False
        result.warnings.append(
            f"Wetland: {profile.current_wetland_pct:.1f}%, Water: {profile.current_water_pct:.1f}% "
            "— wetland areas not eligible for standard ARR (consider Blue Carbon)"
        )
        score -= 5

    # ── Bonus: Degraded land indicators ──
    if profile.current_bare_pct > 20 or profile.mean_ndvi < 0.3:
        result.reasons.append(
            "Degraded land indicators present (bare ground / low vegetation) — "
            "strong candidate for restoration"
        )
        score += 10

    # ── Bonus: Suitable climate for tree growth ──
    if profile.annual_rainfall_mm > 600:
        result.reasons.append(
            f"Annual rainfall ({profile.annual_rainfall_mm:.0f}mm) sufficient for tree establishment"
        )
        score += 5
    else:
        result.warnings.append(
            f"Low rainfall ({profile.annual_rainfall_mm:.0f}mm) — may limit tree species options"
        )

    # ── Size check for small-scale methodology ──
    if profile.total_area_ha <= 8000:
        result.gs_requirements_met["Small-scale threshold (≤8000 ha)"] = True
    else:
        result.gs_requirements_met["Small-scale threshold (≤8000 ha)"] = False
        result.warnings.append("Area exceeds 8000 ha — requires large-scale A/R methodology")

    # Protected area warning
    if profile.in_protected_area:
        result.warnings.append(
            "AOI overlaps with protected area — additional permits and stakeholder engagement required"
        )

    # Final scoring
    score = max(0, min(100, score))
    result.confidence = score

    if score >= 70:
        result.status = EligibilityStatus.HIGHLY_SUITABLE
        result.estimated_potential = "High — strong ARR candidate"
    elif score >= 50:
        result.status = EligibilityStatus.SUITABLE
        result.estimated_potential = "Moderate — ARR feasible with some conditions"
    elif score >= 30:
        result.status = EligibilityStatus.POSSIBLY_SUITABLE
        result.estimated_potential = "Low-Moderate — requires field verification"
    else:
        result.status = EligibilityStatus.NOT_SUITABLE
        result.estimated_potential = "Not recommended for ARR"

    return result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# IFM / REDD+ ELIGIBILITY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def check_ifm_redd_eligibility(profile: LandProfile) -> EligibilityResult:
    """
    Check eligibility for Improved Forest Management (IFM) or REDD+.

    IFM: Land must currently be forested but managed/degraded.
    REDD+: Must demonstrate threat of deforestation (baseline deforestation rate).
    """
    result = EligibilityResult(
        project_type=ProjectType.REDD,
        status=EligibilityStatus.NOT_SUITABLE,
        confidence=0,
        recommended_methodology="GS REDD+ / IFM Requirements"
    )

    score = 0

    # ── Must be currently forested ──
    if profile.current_forest_pct >= 30:
        result.gs_requirements_met["Currently forested (≥30% tree cover)"] = True
        result.reasons.append(
            f"Current forest cover: {profile.current_forest_pct:.1f}% — meets forest threshold"
        )
        score += 25
    else:
        result.gs_requirements_met["Currently forested (≥30% tree cover)"] = False
        result.reasons.append(
            f"Current forest cover only {profile.current_forest_pct:.1f}% — insufficient for IFM/REDD+"
        )
        result.status = EligibilityStatus.NOT_SUITABLE
        result.confidence = max(0, score)
        return result

    # ── Was forested historically ──
    if profile.was_forest_10yr_ago or profile.hansen_forest_cover_2000_pct >= 30:
        result.gs_requirements_met["Historical forest cover confirmed"] = True
        result.reasons.append("Historical forest cover confirmed from Hansen/LULC data")
        score += 15
    else:
        result.gs_requirements_met["Historical forest cover confirmed"] = False
        result.warnings.append("Unable to confirm historical forest status")

    # ── Deforestation pressure (for REDD+) ──
    if profile.has_deforestation or profile.deforestation_rate_pct_yr > 0.1:
        result.gs_requirements_met["Deforestation threat demonstrated"] = True
        result.reasons.append(
            f"Deforestation rate: {profile.deforestation_rate_pct_yr:.2f}%/year "
            f"({profile.deforestation_ha:.1f} ha lost) — demonstrates baseline threat"
        )
        result.project_type = ProjectType.REDD
        score += 25
    else:
        result.gs_requirements_met["Deforestation threat demonstrated"] = False
        result.reasons.append("Low/no deforestation detected — better suited for IFM than REDD+")
        result.project_type = ProjectType.IFM
        score += 10

    # ── Forest degradation indicators ──
    if profile.mean_ndvi < 0.5 and profile.current_forest_pct >= 30:
        result.reasons.append(
            "Relatively low NDVI for forested area — possible degradation, good IFM candidate"
        )
        score += 10

    # ── Forest area size ──
    forest_area_ha = (profile.current_forest_pct / 100) * profile.total_area_ha
    if forest_area_ha >= 50:
        result.reasons.append(f"Forest area: {forest_area_ha:.0f} ha — viable project size")
        score += 15
    else:
        result.warnings.append(f"Forest area only {forest_area_ha:.0f} ha — may be too small")
        score += 5

    # Protected area
    if profile.in_protected_area:
        result.warnings.append("Overlaps protected area — may strengthen conservation case but adds complexity")

    score = max(0, min(100, score))
    result.confidence = score

    if score >= 65:
        result.status = EligibilityStatus.HIGHLY_SUITABLE
        result.estimated_potential = f"High — strong {result.project_type.value} candidate"
    elif score >= 45:
        result.status = EligibilityStatus.SUITABLE
        result.estimated_potential = f"Moderate — {result.project_type.value} feasible"
    elif score >= 25:
        result.status = EligibilityStatus.POSSIBLY_SUITABLE
        result.estimated_potential = "Needs further field assessment"
    else:
        result.status = EligibilityStatus.NOT_SUITABLE
        result.estimated_potential = "Not recommended for IFM/REDD+"

    return result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# BLUE CARBON / MANGROVE ELIGIBILITY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def check_blue_carbon_eligibility(profile: LandProfile) -> EligibilityResult:
    """
    Check eligibility for Blue Carbon / Mangrove project.

    Per GS BCFW Mangrove Methodology (Doc 443, v1.0):
    - Must be in tidal wetland area (mangrove habitat)
    - Historical or current mangrove presence
    - For restoration: evidence of mangrove loss/degradation
    - For conservation: existing mangroves under threat
    - Coastal location with tidal influence
    - Low elevation (typically <10m for mangroves)
    """
    result = EligibilityResult(
        project_type=ProjectType.BLUE_CARBON,
        status=EligibilityStatus.NOT_SUITABLE,
        confidence=0,
        recommended_methodology="GS Sustainable Management of Mangroves (BCFW, Doc 443)"
    )

    score = 0

    # ── Current or historical mangrove presence (CRITICAL) ──
    if profile.mangrove_present:
        result.gs_requirements_met["Mangrove presence confirmed"] = True
        result.reasons.append(
            f"Active mangrove cover detected: {profile.mangrove_area_ha:.1f} ha"
        )
        score += 30
    elif profile.historical_mangrove:
        result.gs_requirements_met["Mangrove presence confirmed"] = True
        result.reasons.append(
            "Historical mangrove presence detected — suitable for mangrove restoration"
        )
        score += 25
    else:
        result.gs_requirements_met["Mangrove presence confirmed"] = False
        result.reasons.append("No current or historical mangrove detected")

    # ── Mangrove loss (for restoration projects) ──
    if profile.mangrove_loss_ha > 0:
        result.gs_requirements_met["Mangrove loss/degradation documented"] = True
        result.reasons.append(
            f"Mangrove loss of {profile.mangrove_loss_ha:.1f} ha documented — "
            "strong restoration opportunity"
        )
        score += 20
    elif profile.historical_mangrove and not profile.mangrove_present:
        result.gs_requirements_met["Mangrove loss/degradation documented"] = True
        result.reasons.append("Complete mangrove loss from historical extent — restoration candidate")
        score += 20

    # ── Tidal wetland / coastal conditions ──
    if profile.tidal_wetland or profile.current_wetland_pct > 5:
        result.gs_requirements_met["Tidal wetland conditions"] = True
        result.reasons.append("Tidal/wetland conditions detected — suitable mangrove habitat")
        score += 15
    elif profile.mean_elevation_m < 15:
        result.gs_requirements_met["Tidal wetland conditions"] = True
        result.reasons.append(
            f"Low elevation ({profile.mean_elevation_m:.0f}m) suggests potential coastal/tidal area"
        )
        score += 10
    else:
        result.gs_requirements_met["Tidal wetland conditions"] = False
        result.warnings.append(
            f"Elevation ({profile.mean_elevation_m:.0f}m) may be too high for mangrove habitat"
        )

    # ── Flooded vegetation indicator ──
    if profile.current_wetland_pct > 10:
        result.reasons.append(
            f"Flooded vegetation covers {profile.current_wetland_pct:.1f}% — "
            "supports tidal wetland classification"
        )
        score += 10

    # ── Climate suitability for mangroves ──
    if "Tropical" in profile.climate_zone:
        result.reasons.append(f"Tropical climate ({profile.climate_zone}) — ideal for mangroves")
        score += 10
    elif "Subtropical" in profile.climate_zone:
        result.reasons.append(f"Subtropical climate — mangroves possible at range limits")
        score += 5

    # ── Soil carbon (mangrove soils typically high in organic carbon) ──
    if profile.soil_carbon_g_per_kg > 20:
        result.reasons.append(
            f"High soil organic carbon ({profile.soil_carbon_g_per_kg:.0f} g/kg) — "
            "indicative of organic-rich coastal soils"
        )
        score += 5

    # Protected area
    if profile.in_protected_area:
        result.warnings.append("Overlaps protected area — may strengthen conservation rationale")

    score = max(0, min(100, score))
    result.confidence = score

    if score >= 60:
        result.status = EligibilityStatus.HIGHLY_SUITABLE
        result.estimated_potential = "High — strong Blue Carbon / Mangrove candidate"
    elif score >= 40:
        result.status = EligibilityStatus.SUITABLE
        result.estimated_potential = "Moderate — Blue Carbon feasible with ground verification"
    elif score >= 20:
        result.status = EligibilityStatus.POSSIBLY_SUITABLE
        result.estimated_potential = "Low — requires coastal/tidal field verification"
    else:
        result.status = EligibilityStatus.NOT_SUITABLE
        result.estimated_potential = "Not recommended for Blue Carbon"

    return result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# AGROFORESTRY ELIGIBILITY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def check_agroforestry_eligibility(profile: LandProfile) -> EligibilityResult:
    """Check eligibility for Agroforestry projects on agricultural land."""
    result = EligibilityResult(
        project_type=ProjectType.AGF,
        status=EligibilityStatus.NOT_SUITABLE,
        confidence=0,
        recommended_methodology="GS A/R with Agroforestry Component"
    )

    score = 0

    # Must be primarily agricultural / cropland
    if profile.current_crop_pct >= 30:
        result.gs_requirements_met["Agricultural land confirmed"] = True
        result.reasons.append(
            f"Cropland covers {profile.current_crop_pct:.1f}% — suitable for agroforestry integration"
        )
        score += 30
    elif profile.current_crop_pct >= 15:
        result.gs_requirements_met["Agricultural land confirmed"] = True
        result.reasons.append(f"Partial agricultural use ({profile.current_crop_pct:.1f}%)")
        score += 15
    else:
        result.gs_requirements_met["Agricultural land confirmed"] = False
        result.reasons.append("Insufficient agricultural land for agroforestry")

    # Low current tree cover (room for tree planting)
    if profile.current_forest_pct < 30:
        result.gs_requirements_met["Low current tree cover"] = True
        result.reasons.append("Low tree cover allows for agroforestry tree integration")
        score += 20
    else:
        result.warnings.append("High existing tree cover — limited scope for additional planting")

    # Not wetland
    if profile.current_wetland_pct < 15:
        score += 10

    # Climate suitability
    if profile.annual_rainfall_mm > 500:
        result.reasons.append(f"Adequate rainfall ({profile.annual_rainfall_mm:.0f}mm) for tree crops")
        score += 10

    # Degradation indicators
    if profile.mean_ndvi < 0.4:
        result.reasons.append("Low vegetation health suggests degraded agricultural land — restoration opportunity")
        score += 10

    score = max(0, min(100, score))
    result.confidence = score

    if score >= 55:
        result.status = EligibilityStatus.SUITABLE
        result.estimated_potential = "Moderate — agroforestry viable"
    elif score >= 35:
        result.status = EligibilityStatus.POSSIBLY_SUITABLE
        result.estimated_potential = "Low-Moderate — needs assessment"
    else:
        result.status = EligibilityStatus.NOT_SUITABLE
        result.estimated_potential = "Not recommended for agroforestry"

    return result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MASTER ELIGIBILITY ASSESSMENT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_eligibility_assessment(analysis_results: Dict) -> Dict:
    """
    Run full eligibility assessment across all project types.
    Returns ranked results with the best-fit project type.
    """
    profile = build_land_profile(analysis_results)

    # Run all eligibility checks
    arr_result = check_arr_eligibility(profile)
    ifm_redd_result = check_ifm_redd_eligibility(profile)
    blue_carbon_result = check_blue_carbon_eligibility(profile)
    agf_result = check_agroforestry_eligibility(profile)

    # Rank by confidence
    all_results = [arr_result, ifm_redd_result, blue_carbon_result, agf_result]
    all_results.sort(key=lambda r: r.confidence, reverse=True)

    # Determine best fit
    best = all_results[0]

    # Build summary
    summary = {
        "land_profile": {
            "total_area_ha": profile.total_area_ha,
            "current_land_use": {
                "forest_pct": round(profile.current_forest_pct, 1),
                "shrub_grassland_pct": round(profile.current_shrub_grass_pct, 1),
                "cropland_pct": round(profile.current_crop_pct, 1),
                "bare_ground_pct": round(profile.current_bare_pct, 1),
                "wetland_pct": round(profile.current_wetland_pct, 1),
                "water_pct": round(profile.current_water_pct, 1),
                "built_pct": round(profile.current_built_pct, 1),
                "mangrove_pct": round(profile.current_mangrove_pct, 1),
            },
            "historical": {
                "was_forest_10yr_ago": profile.was_forest_10yr_ago,
                "forest_pct_10yr_ago": round(profile.forest_pct_10yr_ago, 1),
                "has_deforestation": profile.has_deforestation,
                "deforestation_ha": round(profile.deforestation_ha, 1),
                "hansen_forest_2000_pct": round(profile.hansen_forest_cover_2000_pct, 1),
                "years_non_forest": profile.years_non_forest,
            },
            "mangrove": {
                "present": profile.mangrove_present,
                "area_ha": round(profile.mangrove_area_ha, 1),
                "loss_ha": round(profile.mangrove_loss_ha, 1),
                "historical": profile.historical_mangrove,
                "tidal_wetland": profile.tidal_wetland,
            },
            "environment": {
                "climate_zone": profile.climate_zone,
                "rainfall_mm": round(profile.annual_rainfall_mm, 0),
                "elevation_m": round(profile.mean_elevation_m, 0),
                "slope_deg": round(profile.mean_slope_deg, 1),
                "mean_ndvi": round(profile.mean_ndvi, 3),
                "soil_carbon_g_per_kg": round(profile.soil_carbon_g_per_kg, 1),
                "in_protected_area": profile.in_protected_area,
            },
        },
        "recommended_project_type": best.project_type.value,
        "recommended_methodology": best.recommended_methodology,
        "overall_confidence": best.confidence,
        "results": [],
    }

    for r in all_results:
        summary["results"].append({
            "project_type": r.project_type.value,
            "status": r.status.value,
            "confidence": r.confidence,
            "reasons": r.reasons,
            "warnings": r.warnings,
            "gs_requirements_met": r.gs_requirements_met,
            "methodology": r.recommended_methodology,
            "estimated_potential": r.estimated_potential,
        })

    # ── Generate overall recommendation narrative ──
    summary["narrative"] = generate_recommendation_narrative(profile, all_results)

    return summary


def generate_recommendation_narrative(profile: LandProfile, results: List[EligibilityResult]) -> str:
    """Generate a human-readable recommendation summary."""
    best = results[0]

    narrative = f"Based on the spatial analysis of {profile.total_area_ha:.1f} hectares "
    narrative += f"located at approximately {profile.climate_zone} climate zone, "

    if best.status in [EligibilityStatus.HIGHLY_SUITABLE, EligibilityStatus.SUITABLE]:
        narrative += f"the land is **{best.status.value}** for **{best.project_type.value}** projects. "
    elif best.status == EligibilityStatus.POSSIBLY_SUITABLE:
        narrative += f"the land **may be suitable** for **{best.project_type.value}** projects, "
        narrative += "pending field verification. "
    else:
        narrative += "the land does not appear well-suited for any carbon project type "
        narrative += "based on current spatial data. "

    # Key findings
    narrative += "\n\n**Key Findings:**\n"
    if profile.current_forest_pct > 30:
        narrative += f"- Currently forested ({profile.current_forest_pct:.0f}% tree cover)\n"
    else:
        narrative += f"- Currently non-forest ({profile.current_forest_pct:.0f}% tree cover)\n"

    if profile.has_deforestation:
        narrative += f"- Historical deforestation detected ({profile.deforestation_ha:.1f} ha)\n"
    else:
        narrative += "- No significant deforestation detected\n"

    if profile.mangrove_present or profile.historical_mangrove:
        narrative += f"- Mangrove presence: {'Current' if profile.mangrove_present else 'Historical only'}\n"

    narrative += f"- Mean NDVI: {profile.mean_ndvi:.3f} "
    if profile.mean_ndvi > 0.5:
        narrative += "(healthy vegetation)\n"
    elif profile.mean_ndvi > 0.3:
        narrative += "(moderate vegetation)\n"
    else:
        narrative += "(sparse/degraded vegetation)\n"

    # Recommendations
    narrative += "\n**Recommended Next Steps:**\n"
    narrative += "1. Conduct field verification visit to ground-truth satellite classifications\n"
    narrative += "2. Assess land ownership and community engagement requirements\n"
    narrative += f"3. Develop detailed Project Design Document (PDD) under {best.recommended_methodology}\n"
    narrative += "4. Engage Gold Standard-approved Validation & Verification Body (VVB)\n"

    return narrative
