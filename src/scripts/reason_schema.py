"""Versioned visual reason names shared by labeling, training and inference."""

REASON_SCHEMA_VERSION = 2
REASONS = ("corrosion", "mechanical_damage", "heavy_contamination", "wrong_brand_or_type")
LEGACY_REASONS = ("major_damage", "heavy_dirt", "wrong_brand_or_model")
PROJECT_POLICY = "primagaz_only_v1"

REASON_LABELS = {
    "corrosion": "Significant corrosion",
    "mechanical_damage": "Major mechanical damage",
    "heavy_contamination": "Heavy contamination",
    "wrong_brand_or_type": "Wrong brand or bottle type",
}


def read_reason_schema(metadata, allow_legacy=False):
    """Validate meanings and order; legacy outputs are never renamed."""
    version = metadata.get("reason_schema_version")
    raw_order = metadata.get("reason_order")
    order = tuple(raw_order) if isinstance(raw_order, (list, tuple)) else ()
    if (type(version) is int and version == REASON_SCHEMA_VERSION
            and order == REASONS and metadata.get("project_policy") == PROJECT_POLICY):
        return version, order, PROJECT_POLICY
    if (allow_legacy and (version is None or type(version) is int and version == 1)
            and order == LEGACY_REASONS and metadata.get("project_policy") is None):
        return 1, order, None
    raise ValueError(
        "Expected reason schema v2 with the four canonical reasons and "
        "project_policy='primagaz_only_v1'. Legacy three-reason data needs human "
        "re-review; old major_damage labels cannot be renamed or padded into v2."
    )
