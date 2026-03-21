"""Centralised filename constants for the energy estimation and profiling pipeline."""

# Written by CodeCarbon's EmissionsTracker
EMISSIONS_CSV_FILENAME = "emissions.csv"

# Written by EstimationPipeline (pre-run estimates)
EMISSION_ESTIMATE_FILENAME = "emission_estimate.json"

# Written by EmissionComparator (post-run comparison)
EMISSION_COMPARISON_FILENAME = "emission_comparison.json"

# Default carbon intensity used for CO₂ estimation (grams CO₂ per kWh).
# Source: https://www.nowtricity.com/country/germany/ (average for Germany, 2025)
DEFAULT_CARBON_INTENSITY_G_PER_KWH = 328
