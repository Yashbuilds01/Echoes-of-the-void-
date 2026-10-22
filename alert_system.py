"""
Echoes of the Void - Multi-Messenger Alert Dispatcher Module
Generates automated GCN Circulars and IVOA-compliant JSON notices
when cosmic gravitational-wave transients are detected with >85% confidence.
"""

import json
import numpy as np
from datetime import datetime, timezone
from typing import Dict, Any, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("EchoesOfTheVoid.AlertSystem")


class AlertSystem:
    """
    Automated Multi-Messenger Alert Coordinator.
    Transforms high-confidence AI triggers into global astronomical notices
    with celestial coordinates, sky localization ellipses, and telescope pointing schedules.
    """

    CONFIDENCE_THRESHOLD = 0.85  # 85% confidence trigger criterion

    @classmethod
    def evaluate_and_dispatch(
        cls,
        classification_result: Dict[str, Any],
        source_name: str,
        snr: float,
        event_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Evaluates AI classification confidence and generates an astronomical alert package
        if the event is a genuine cosmic trigger above threshold.
        """
        pred_class = classification_result.get("predicted_class", "Instrumental Glitch / Background Noise")
        confidence = classification_result.get("confidence", 0.0)

        # Glitches and background noise do NOT trigger cosmic sky alerts
        is_cosmic = ("Black Hole" in pred_class or "Neutron Star" in pred_class or "Supernova" in pred_class)
        triggered = is_cosmic and (confidence >= cls.CONFIDENCE_THRESHOLD)

        now_utc = datetime.now(timezone.utc)
        timestamp_str = now_utc.strftime("%Y-%m-%d %H:%M:%S UTC")
        alert_id = f"ECHO-GW{now_utc.strftime('%y%m%d_%H%M%S')}"

        if not triggered:
            return {
                "triggered": False,
                "reason": (
                    f"Confidence {confidence*100:.1f}% below {cls.CONFIDENCE_THRESHOLD*100:.0f}% threshold"
                    if is_cosmic else f"Classified as non-cosmic artifact ({pred_class})"
                ),
                "alert_id": alert_id,
                "timestamp": timestamp_str
            }

        # Synthesize realistic celestial coordinates and localization
        ra_deg, dec_deg, ra_hms, dec_dms = cls._generate_celestial_coords(pred_class)
        sky_area_sq_deg = round(float(np.random.uniform(18.0, 48.0) if snr > 20 else np.random.uniform(85.0, 240.0)), 1)
        far_years = round(float(np.random.uniform(1.2e4, 8.5e6)), 1)
        
        # Estimate luminosity distance based on event type
        if "Neutron Star" in pred_class:
            dist_mpc = round(float(np.random.uniform(35.0, 85.0)), 1)
            dist_err = round(dist_mpc * 0.22, 1)
            source_abbr = "BNS"
        elif "Supernova" in pred_class:
            dist_mpc = round(float(np.random.uniform(5.0, 30.0)), 1)
            dist_err = round(dist_mpc * 0.35, 1)
            source_abbr = "Supernova"
        else:
            dist_mpc = round(float(np.random.uniform(380.0, 1200.0)), 1)
            dist_err = round(dist_mpc * 0.25, 1)
            source_abbr = "BBH"

        # Construct JSON notice
        json_payload = {
            "$schema": "https://gcn.nasa.gov/schema/v4/gcn/notices/lvk/alert.schema.json",
            "alert_type": "INITIAL",
            "alert_id": alert_id,
            "pipeline": "Echoes of the Void (Real-Time PyTorch AI & BIC Pipeline)",
            "author": "Yash Goyal & Vidushi Sharma (GLA University) / Echoes of the Void Automated Broker",
            "time_utc": timestamp_str,
            "trigger_classification": {
                "primary": pred_class,
                "confidence": round(confidence, 4),
                "class_probabilities": classification_result.get("probabilities", {})
            },
            "astrophysical_parameters": {
                "network_snr": round(float(snr), 2),
                "false_alarm_rate_yr": f"1 per {far_years:,.0f} years",
                "luminosity_distance_mpc": f"{dist_mpc} ± {dist_err}",
                "chirp_slope_hz_s": classification_result.get("chirp_slope_hz_s", 0.0),
                "inference_latency_ms": classification_result.get("latency_ms", 12.0)
            },
            "sky_localization": {
                "ra_degrees": round(ra_deg, 4),
                "dec_degrees": round(dec_deg, 4),
                "ra_hms": ra_hms,
                "dec_dms": dec_dms,
                "area_90_sq_deg": sky_area_sq_deg,
                "search_cone_radius_deg": round(np.sqrt(sky_area_sq_deg / np.pi), 2)
            },
            "recommended_followup": [
                {"facility": "ZTF (Zwicky Transient Facility)", "priority": "CRITICAL", "bands": "g, r, i deep tiles"},
                {"facility": "Vera C. Rubin Observatory (LSST)", "priority": "HIGH", "bands": "u, g, r, i, z, y"},
                {"facility": "NASA Swift-XRT", "priority": "RAPID", "mode": "Target of Opportunity (ToO) X-ray tiling"},
                {"facility": "Nancy Grace Roman Space Telescope", "priority": "PLANNED", "mode": "NIR Wide-field counterpart search"}
            ]
        }

        # Construct human-readable GCN Circular text
        gcn_text = cls._generate_gcn_circular(
            alert_id=alert_id,
            timestamp=timestamp_str,
            pred_class=pred_class,
            confidence=confidence,
            snr=snr,
            ra_hms=ra_hms,
            dec_dms=dec_dms,
            ra_deg=ra_deg,
            dec_deg=dec_deg,
            area_deg=sky_area_sq_deg,
            dist_mpc=dist_mpc,
            dist_err=dist_err,
            far_years=far_years
        )

        return {
            "triggered": True,
            "alert_id": alert_id,
            "timestamp": timestamp_str,
            "pred_class": pred_class,
            "confidence": confidence,
            "snr": snr,
            "ra_deg": ra_deg,
            "dec_deg": dec_deg,
            "ra_hms": ra_hms,
            "dec_dms": dec_dms,
            "area_90_sq_deg": sky_area_sq_deg,
            "dist_mpc": dist_mpc,
            "dist_err": dist_err,
            "json_payload": json_payload,
            "gcn_circular": gcn_text
        }

    @staticmethod
    def _generate_celestial_coords(event_type: str) -> Tuple[float, float, str, str]:
        """Generates mock astronomical celestial coordinates (RA/Dec)."""
        ra_deg = float(np.random.uniform(15.0, 350.0))
        dec_deg = float(np.random.uniform(-65.0, 65.0))

        # Convert RA to Hours, Minutes, Seconds
        ra_hours_total = ra_deg / 15.0
        ra_h = int(ra_hours_total)
        ra_m = int((ra_hours_total - ra_h) * 60)
        ra_s = (ra_hours_total - ra_h - ra_m / 60.0) * 3600.0
        ra_hms = f"{ra_h:02d}h {ra_m:02d}m {ra_s:04.1f}s"

        # Convert Dec to Degrees, Arcminutes, Arcseconds
        sign = "+" if dec_deg >= 0 else "-"
        abs_dec = abs(dec_deg)
        dec_d = int(abs_dec)
        dec_m = int((abs_dec - dec_d) * 60)
        dec_s = (abs_dec - dec_d - dec_m / 60.0) * 3600.0
        dec_dms = f"{sign}{dec_d:02d}° {dec_m:02d}' {dec_s:04.1f}\""

        return ra_deg, dec_deg, ra_hms, dec_dms

    @staticmethod
    def _generate_gcn_circular(
        alert_id: str,
        timestamp: str,
        pred_class: str,
        confidence: float,
        snr: float,
        ra_hms: str,
        dec_dms: str,
        ra_deg: float,
        dec_deg: float,
        area_deg: float,
        dist_mpc: float,
        dist_err: float,
        far_years: float
    ) -> str:
        """Formats an official GCN (Gamma-ray Coordinates Network) Circular."""
        return f"""TITLE:   GCN CIRCULAR
NUMBER:  {alert_id}
SUBJECT: Echoes of the Void Real-Time Discovery: Gravitational-Wave Transient Detection
DATE:    {timestamp}
FROM:    Yash Goyal & Vidushi Sharma on behalf of the Echoes of the Void Collaboration (GLA University)

The 'Echoes of the Void' real-time machine intelligence pipeline identified a 
high-significance gravitational wave candidate:

Trigger ID:              {alert_id}
Event Classification:    {pred_class} (Confidence: {confidence*100:.2f}%)
Network SNR:             {snr:.2f}
False Alarm Rate:        1 per {far_years:,.0f} years
Estimated Distance:      {dist_mpc} ± {dist_err} Mpc

SKY LOCALIZATION (J2000):
Central Coordinates:     RA = {ra_hms} ({ra_deg:.4f} deg)
                         Dec = {dec_dms} ({dec_deg:.4f} deg)
90% Localization Area:   {area_deg} sq deg

RECOMMENDED MULTI-MESSENGER OBSERVING STRATEGY:
Optical and X-ray facilities (ZTF, Rubin/LSST, Swift-XRT) are strongly urged 
to initiate rapid target-of-opportunity tiling across the 90% credible contour.
Particular focus should be given to host galaxy catalogs within {dist_mpc} Mpc.

Further automated updates and prompt kilonova/afterglow candidate screenings 
will be broadcast through the Echoes of the Void broker stream.
"""
