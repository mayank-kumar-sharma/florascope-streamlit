"""
Project Audit Engine
Comprehensive spatial audit for forestry carbon projects.

Checks:
  1. Farm-in-project containment (farms within project boundary)
  2. Area outside project boundary flagging
  3. Boundary overlap detection between farms
  4. Duplicate farm detection
  5. Self-intersecting polygon detection
  6. Buffer zone compliance (15m water body buffers per GS LUF)
  7. Gap analysis (uncovered areas within project)
  8. Total vs eligible area reconciliation
  9. Geometry validity checks
  10. Area consistency verification
"""

import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from shapely.geometry import shape, mapping, Polygon, MultiPolygon
from shapely.ops import unary_union
from shapely.validation import explain_validity
import pyproj
from shapely.ops import transform as shapely_transform
from functools import partial

from .geometry_tools import (
    calculate_area, reproject_geometry, get_utm_crs, _extract_geometry
)


# ──────────────────────────────────────────────
# DATA CLASSES
# ──────────────────────────────────────────────

class SeverityLevel:
    CRITICAL = "critical"    # Blocks certification
    WARNING = "warning"      # Needs attention
    INFO = "info"           # Informational
    PASS = "pass"           # Check passed


@dataclass
class AuditIssue:
    check_name: str
    severity: str
    message: str
    affected_features: List[int] = field(default_factory=list)
    geometry: Optional[dict] = None  # GeoJSON of problem area
    area_ha: float = 0
    details: Dict = field(default_factory=dict)

    def to_dict(self):
        return {
            "check_name": self.check_name,
            "severity": self.severity,
            "message": self.message,
            "affected_features": self.affected_features,
            "geometry": self.geometry,
            "area_ha": round(self.area_ha, 4),
            "details": self.details,
        }


@dataclass
class AuditReport:
    project_name: str = ""
    total_issues: int = 0
    critical_count: int = 0
    warning_count: int = 0
    info_count: int = 0
    pass_count: int = 0
    issues: List[AuditIssue] = field(default_factory=list)
    summary: Dict = field(default_factory=dict)

    def add_issue(self, issue: AuditIssue):
        self.issues.append(issue)
        self.total_issues += 1
        if issue.severity == SeverityLevel.CRITICAL:
            self.critical_count += 1
        elif issue.severity == SeverityLevel.WARNING:
            self.warning_count += 1
        elif issue.severity == SeverityLevel.INFO:
            self.info_count += 1
        else:
            self.pass_count += 1

    def to_dict(self):
        return {
            "project_name": self.project_name,
            "total_issues": self.total_issues,
            "critical_count": self.critical_count,
            "warning_count": self.warning_count,
            "info_count": self.info_count,
            "pass_count": self.pass_count,
            "overall_status": self._overall_status(),
            "issues": [i.to_dict() for i in self.issues],
            "summary": self.summary,
        }

    def _overall_status(self):
        if self.critical_count > 0:
            return "FAIL"
        elif self.warning_count > 0:
            return "NEEDS_ATTENTION"
        else:
            return "PASS"


# ──────────────────────────────────────────────
# HELPER: ACCURATE AREA IN HECTARES
# ──────────────────────────────────────────────

def _area_ha(geom) -> float:
    """Get area in hectares using UTM projection."""
    if geom is None or geom.is_empty:
        return 0
    try:
        geom_utm = reproject_geometry(geom)
        return geom_utm.area / 10000
    except Exception:
        # Fallback: rough calculation
        return geom.area * (111000 ** 2) / 10000


# ──────────────────────────────────────────────
# MAIN AUDIT FUNCTION
# ──────────────────────────────────────────────

def run_full_audit(
    project_boundary_geojson: dict,
    farm_plots_geojson: dict,
    project_name: str = "Project",
    buffer_distance_m: float = 15.0,
    overlap_threshold_pct: float = 5.0,
    duplicate_area_threshold_pct: float = 90.0,
) -> Dict:
    """
    Run comprehensive spatial audit.

    Args:
        project_boundary_geojson: GeoJSON of the project boundary
        farm_plots_geojson: GeoJSON FeatureCollection of farm/plot polygons
        project_name: Name for the report
        buffer_distance_m: Buffer zone distance (default 15m per GS LUF)
        overlap_threshold_pct: % overlap to flag between farms
        duplicate_area_threshold_pct: % overlap to consider duplicate
    """
    report = AuditReport(project_name=project_name)

    # Parse geometries
    project_geom = _extract_geometry(project_boundary_geojson)
    if project_geom is None:
        return {"error": "Invalid project boundary geometry"}

    if not project_geom.is_valid:
        project_geom = project_geom.buffer(0)

    farms = []
    farm_features = farm_plots_geojson.get("features", [])
    for i, feat in enumerate(farm_features):
        try:
            geom = shape(feat["geometry"])
            if not geom.is_valid:
                geom = geom.buffer(0)
            name = feat.get("properties", {}).get("name",
                   feat.get("properties", {}).get("Name", f"Farm {i+1}"))
            farms.append({"index": i, "name": name, "geom": geom, "feature": feat})
        except Exception as e:
            report.add_issue(AuditIssue(
                check_name="Geometry Parsing",
                severity=SeverityLevel.CRITICAL,
                message=f"Farm {i+1}: Failed to parse geometry — {str(e)}",
                affected_features=[i]
            ))

    # Project area
    project_area_ha = _area_ha(project_geom)

    # ── Run all checks ──
    _check_geometry_validity(report, farms)
    _check_self_intersections(report, farms)
    _check_containment(report, project_geom, farms, project_area_ha)
    _check_farm_overlaps(report, farms, overlap_threshold_pct)
    _check_duplicates(report, farms, duplicate_area_threshold_pct)
    _check_gaps(report, project_geom, farms, project_area_ha)
    _check_area_consistency(report, project_geom, farms, project_area_ha)

    # ── Build summary ──
    total_farm_area = sum(_area_ha(f["geom"]) for f in farms)
    report.summary = {
        "project_area_ha": round(project_area_ha, 4),
        "total_farm_count": len(farms),
        "total_farm_area_ha": round(total_farm_area, 4),
        "coverage_pct": round((total_farm_area / max(project_area_ha, 0.001)) * 100, 1),
        "farms_fully_inside": sum(1 for f in farms if project_geom.contains(f["geom"])),
        "farms_partially_outside": sum(
            1 for f in farms
            if not project_geom.contains(f["geom"]) and project_geom.intersects(f["geom"])
        ),
        "farms_fully_outside": sum(
            1 for f in farms if not project_geom.intersects(f["geom"])
        ),
    }

    return report.to_dict()


# ──────────────────────────────────────────────
# CHECK 1: GEOMETRY VALIDITY
# ──────────────────────────────────────────────

def _check_geometry_validity(report: AuditReport, farms: List[Dict]):
    """Check all farm polygons for geometry validity."""
    invalid_count = 0
    for farm in farms:
        geom = farm["geom"]
        if not geom.is_valid:
            reason = explain_validity(geom)
            report.add_issue(AuditIssue(
                check_name="Geometry Validity",
                severity=SeverityLevel.WARNING,
                message=f"{farm['name']}: Invalid geometry — {reason}",
                affected_features=[farm["index"]],
                details={"reason": reason}
            ))
            invalid_count += 1

    if invalid_count == 0:
        report.add_issue(AuditIssue(
            check_name="Geometry Validity",
            severity=SeverityLevel.PASS,
            message=f"All {len(farms)} farm geometries are valid",
        ))


# ──────────────────────────────────────────────
# CHECK 2: SELF-INTERSECTIONS
# ──────────────────────────────────────────────

def _check_self_intersections(report: AuditReport, farms: List[Dict]):
    """Detect self-intersecting polygons (bowtie shapes)."""
    count = 0
    for farm in farms:
        geom = farm["geom"]
        if hasattr(geom, "exterior"):
            ring = geom.exterior
            if not ring.is_simple:
                report.add_issue(AuditIssue(
                    check_name="Self-Intersection",
                    severity=SeverityLevel.CRITICAL,
                    message=f"{farm['name']}: Polygon has self-intersecting boundary (bowtie/figure-8 shape)",
                    affected_features=[farm["index"]],
                ))
                count += 1

    if count == 0:
        report.add_issue(AuditIssue(
            check_name="Self-Intersection",
            severity=SeverityLevel.PASS,
            message="No self-intersecting polygons found",
        ))


# ──────────────────────────────────────────────
# CHECK 3: CONTAINMENT (Farm in Project)
# ──────────────────────────────────────────────

def _check_containment(report: AuditReport, project_geom, farms: List[Dict], project_area_ha: float):
    """Check if each farm is fully within the project boundary."""
    fully_inside = 0
    partial_outside = 0
    fully_outside = 0

    for farm in farms:
        geom = farm["geom"]
        farm_area = _area_ha(geom)

        if project_geom.contains(geom):
            fully_inside += 1
        elif project_geom.intersects(geom):
            # Partially outside
            partial_outside += 1
            outside_geom = geom.difference(project_geom)
            inside_geom = geom.intersection(project_geom)
            outside_area = _area_ha(outside_geom)
            inside_area = _area_ha(inside_geom)
            outside_pct = (outside_area / max(farm_area, 0.0001)) * 100

            report.add_issue(AuditIssue(
                check_name="Farm Containment",
                severity=SeverityLevel.CRITICAL if outside_pct > 10 else SeverityLevel.WARNING,
                message=(
                    f"{farm['name']}: {outside_pct:.1f}% of farm area ({outside_area:.4f} ha) "
                    f"is OUTSIDE the project boundary"
                ),
                affected_features=[farm["index"]],
                geometry=mapping(outside_geom) if not outside_geom.is_empty else None,
                area_ha=outside_area,
                details={
                    "farm_area_ha": round(farm_area, 4),
                    "inside_area_ha": round(inside_area, 4),
                    "outside_area_ha": round(outside_area, 4),
                    "outside_pct": round(outside_pct, 1),
                }
            ))
        else:
            # Fully outside
            fully_outside += 1
            report.add_issue(AuditIssue(
                check_name="Farm Containment",
                severity=SeverityLevel.CRITICAL,
                message=f"{farm['name']}: Farm is ENTIRELY OUTSIDE the project boundary ({farm_area:.4f} ha)",
                affected_features=[farm["index"]],
                area_ha=farm_area,
            ))

    if fully_inside == len(farms) and len(farms) > 0:
        report.add_issue(AuditIssue(
            check_name="Farm Containment",
            severity=SeverityLevel.PASS,
            message=f"All {len(farms)} farms are fully within the project boundary",
        ))


# ──────────────────────────────────────────────
# CHECK 4: FARM OVERLAPS
# ──────────────────────────────────────────────

def _check_farm_overlaps(report: AuditReport, farms: List[Dict], threshold_pct: float):
    """Detect overlapping farm boundaries."""
    overlap_count = 0
    checked_pairs = set()

    for i, farm_a in enumerate(farms):
        for j, farm_b in enumerate(farms):
            if i >= j:
                continue
            pair_key = (min(i, j), max(i, j))
            if pair_key in checked_pairs:
                continue
            checked_pairs.add(pair_key)

            if farm_a["geom"].intersects(farm_b["geom"]):
                intersection = farm_a["geom"].intersection(farm_b["geom"])
                if intersection.is_empty or intersection.geom_type in ("Point", "LineString", "MultiPoint", "MultiLineString"):
                    continue  # Just touching, not overlapping area

                overlap_area = _area_ha(intersection)
                smaller_area = min(_area_ha(farm_a["geom"]), _area_ha(farm_b["geom"]))
                overlap_pct = (overlap_area / max(smaller_area, 0.0001)) * 100

                if overlap_pct > threshold_pct:
                    overlap_count += 1
                    report.add_issue(AuditIssue(
                        check_name="Farm Overlap",
                        severity=SeverityLevel.CRITICAL if overlap_pct > 50 else SeverityLevel.WARNING,
                        message=(
                            f"{farm_a['name']} ↔ {farm_b['name']}: "
                            f"Overlap of {overlap_area:.4f} ha ({overlap_pct:.1f}% of smaller farm)"
                        ),
                        affected_features=[farm_a["index"], farm_b["index"]],
                        geometry=mapping(intersection),
                        area_ha=overlap_area,
                        details={
                            "farm_a": farm_a["name"],
                            "farm_b": farm_b["name"],
                            "overlap_pct": round(overlap_pct, 1),
                        }
                    ))

    if overlap_count == 0:
        report.add_issue(AuditIssue(
            check_name="Farm Overlap",
            severity=SeverityLevel.PASS,
            message=f"No significant overlaps detected among {len(farms)} farms",
        ))


# ──────────────────────────────────────────────
# CHECK 5: DUPLICATE DETECTION
# ──────────────────────────────────────────────

def _check_duplicates(report: AuditReport, farms: List[Dict], threshold_pct: float):
    """Detect near-duplicate farm polygons."""
    duplicates = []
    checked = set()

    for i, farm_a in enumerate(farms):
        for j, farm_b in enumerate(farms):
            if i >= j:
                continue
            pair = (i, j)
            if pair in checked:
                continue
            checked.add(pair)

            area_a = _area_ha(farm_a["geom"])
            area_b = _area_ha(farm_b["geom"])

            # Check if areas are similar (within 20%)
            if max(area_a, area_b) == 0:
                continue
            area_ratio = min(area_a, area_b) / max(area_a, area_b) * 100

            if area_ratio > 80:  # Similar size
                intersection = farm_a["geom"].intersection(farm_b["geom"])
                overlap_area = _area_ha(intersection)
                overlap_pct = (overlap_area / max(min(area_a, area_b), 0.0001)) * 100

                if overlap_pct >= threshold_pct:
                    duplicates.append((i, j))
                    report.add_issue(AuditIssue(
                        check_name="Duplicate Detection",
                        severity=SeverityLevel.CRITICAL,
                        message=(
                            f"Possible DUPLICATE: {farm_a['name']} and {farm_b['name']} "
                            f"overlap by {overlap_pct:.1f}% with similar area"
                        ),
                        affected_features=[i, j],
                        details={
                            "overlap_pct": round(overlap_pct, 1),
                            "area_a_ha": round(area_a, 4),
                            "area_b_ha": round(area_b, 4),
                        }
                    ))

    if not duplicates:
        report.add_issue(AuditIssue(
            check_name="Duplicate Detection",
            severity=SeverityLevel.PASS,
            message="No duplicate farms detected",
        ))


# ──────────────────────────────────────────────
# CHECK 6: GAP ANALYSIS
# ──────────────────────────────────────────────

def _check_gaps(report: AuditReport, project_geom, farms: List[Dict], project_area_ha: float):
    """Analyze uncovered gaps between farms within the project boundary."""
    if not farms:
        return

    try:
        all_farms_union = unary_union([f["geom"] for f in farms])
        # Clip farms to project boundary first
        farms_in_project = all_farms_union.intersection(project_geom)
        gap_geom = project_geom.difference(farms_in_project)

        gap_area = _area_ha(gap_geom)
        gap_pct = (gap_area / max(project_area_ha, 0.001)) * 100

        if gap_pct > 30:
            severity = SeverityLevel.INFO
        elif gap_pct > 0:
            severity = SeverityLevel.INFO
        else:
            severity = SeverityLevel.PASS

        report.add_issue(AuditIssue(
            check_name="Gap Analysis",
            severity=severity,
            message=(
                f"Uncovered area within project boundary: {gap_area:.2f} ha ({gap_pct:.1f}%). "
                f"This may include infrastructure, water bodies, buffer zones, or non-eligible areas."
            ),
            geometry=mapping(gap_geom) if not gap_geom.is_empty and gap_area > 0 else None,
            area_ha=gap_area,
            details={"gap_pct": round(gap_pct, 1)}
        ))
    except Exception as e:
        report.add_issue(AuditIssue(
            check_name="Gap Analysis",
            severity=SeverityLevel.WARNING,
            message=f"Could not compute gap analysis: {str(e)}",
        ))


# ──────────────────────────────────────────────
# CHECK 7: AREA CONSISTENCY
# ──────────────────────────────────────────────

def _check_area_consistency(report: AuditReport, project_geom, farms: List[Dict], project_area_ha: float):
    """Check that total farm area doesn't exceed project area (double-counting)."""
    total_farm_area = sum(_area_ha(f["geom"]) for f in farms)
    total_farm_in_project = 0

    for farm in farms:
        clipped = farm["geom"].intersection(project_geom)
        total_farm_in_project += _area_ha(clipped)

    # Check for double-counting (overlapping farms create total > project)
    if total_farm_area > project_area_ha * 1.05:  # 5% tolerance
        excess = total_farm_area - project_area_ha
        report.add_issue(AuditIssue(
            check_name="Area Consistency",
            severity=SeverityLevel.WARNING,
            message=(
                f"Total farm area ({total_farm_area:.2f} ha) exceeds project area "
                f"({project_area_ha:.2f} ha) by {excess:.2f} ha — "
                f"likely due to overlapping farms or farms extending outside boundary"
            ),
            details={
                "total_farm_ha": round(total_farm_area, 4),
                "project_ha": round(project_area_ha, 4),
                "excess_ha": round(excess, 4),
            }
        ))
    else:
        report.add_issue(AuditIssue(
            check_name="Area Consistency",
            severity=SeverityLevel.PASS,
            message=(
                f"Area check passed: Farm area ({total_farm_area:.2f} ha) ≤ "
                f"Project area ({project_area_ha:.2f} ha)"
            ),
        ))

    # Eligible area summary
    report.summary["eligible_area_ha"] = round(total_farm_in_project, 4)
    report.summary["non_eligible_area_ha"] = round(
        project_area_ha - total_farm_in_project, 4
    )


# ──────────────────────────────────────────────
# SINGLE FARM VALIDATION
# ──────────────────────────────────────────────

def validate_single_farm(farm_geojson: dict, project_boundary_geojson: Optional[dict] = None) -> Dict:
    """
    Quick validation of a single farm polygon.
    Returns validation results and area info.
    """
    geom = _extract_geometry(farm_geojson)
    if geom is None:
        return {"valid": False, "error": "Could not parse geometry"}

    result = {
        "valid": geom.is_valid,
        "validity_reason": explain_validity(geom) if not geom.is_valid else "Valid Geometry",
        "geometry_type": geom.geom_type,
        "area_ha": round(_area_ha(geom), 4),
        "vertices": len(geom.exterior.coords) if hasattr(geom, "exterior") else 0,
        "is_simple": geom.is_simple if hasattr(geom, "is_simple") else True,
        "centroid": {"lat": round(geom.centroid.y, 6), "lon": round(geom.centroid.x, 6)},
    }

    if project_boundary_geojson:
        proj_geom = _extract_geometry(project_boundary_geojson)
        if proj_geom:
            result["in_project"] = proj_geom.contains(geom)
            result["intersects_project"] = proj_geom.intersects(geom)

            if not result["in_project"] and result["intersects_project"]:
                outside = geom.difference(proj_geom)
                result["area_outside_ha"] = round(_area_ha(outside), 4)
                result["area_outside_pct"] = round(
                    (_area_ha(outside) / max(_area_ha(geom), 0.0001)) * 100, 1
                )
                result["outside_geometry"] = mapping(outside) if not outside.is_empty else None
            elif not result["intersects_project"]:
                result["area_outside_ha"] = result["area_ha"]
                result["area_outside_pct"] = 100.0

    return result
