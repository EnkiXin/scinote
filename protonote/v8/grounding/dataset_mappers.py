"""Map raw dataset annotations to unified scinote V8 label schema.

Each dataset uses its own naming. We normalize all of them into:
  (label, entity_type)
where `entity_type` is one of v8.kg.entity.EntityType.

Maps are written for what's actually in the data (verified from
data.yaml / annotation files), not from a paper abstract.
"""

# ============================================================
# ChemEq25 — 25 classes from data.yaml (verified on disk)
# ============================================================
#
# Source: figshare 29110433 v3 ("ChemistryLabApparatus-25").
# Total 4,599 images (3220 train / 920 valid / 459 test), YOLO labels.
# Classes are exact strings from data.yaml `names:` field.
CHEMEQ25_MAP = {
    "Beaker":                                          ("beaker",                       "Container"),
    "Buchner_Funnel":                                  ("Buchner funnel",               "Container"),
    "Burette_Stands":                                  ("burette stand",                "Instrument"),
    "Calorimeter":                                     ("calorimeter",                  "Instrument"),
    "Conical_Flask":                                   ("Erlenmeyer flask",             "Container"),
    "Funnel":                                          ("funnel",                       "Container"),
    "Glass_Rod":                                       ("glass stirring rod",           "Instrument"),
    "Measuring_Cylinder":                              ("graduated cylinder",           "Container"),
    "Mechanical_Balance_Scale":                        ("mechanical balance",           "Instrument"),
    "Nessler_Reagent_Bottle":                          ("Nessler reagent bottle",       "Container"),
    "Pipette":                                         ("pipette",                      "Instrument"),
    "Porcelain_Mortar Pestle":                         ("mortar and pestle",            "Instrument"),
    "Precision_Weight_Scale":                          ("precision balance",            "Instrument"),
    "Reagent_Bottle":                                  ("reagent bottle",               "Container"),
    "Round_Bottom_Flask_Borosilicate_Glass_1_Neck":    ("round-bottom flask, 1-neck",   "Container"),
    "Round_Bottom_Flask_Borosilicate_Glass_2_Neck":    ("round-bottom flask, 2-neck",   "Container"),
    "Round_Bottom_Flask_Borosilicate_Glass_3_Neck":    ("round-bottom flask, 3-neck",   "Container"),
    "Separating_Funnel":                               ("separating funnel",            "Container"),
    "Spirit_Lamp":                                     ("spirit lamp",                  "Instrument"),
    "TestTube_Holder":                                 ("test tube holder",             "Instrument"),
    "Test_Tube":                                       ("test tube",                    "Container"),
    "Volumetric_Flask":                                ("volumetric flask",             "Container"),
    "Volumetric_Pipet":                                ("volumetric pipette",           "Instrument"),
    "Wash_Bottle":                                     ("wash bottle",                  "Container"),
    "Weighing_Bottle":                                 ("weighing bottle",              "Container"),
}


# ============================================================
# Vector-LabPics V2 — vessels + materials
# ============================================================
#
# Source: Zenodo 4736111 (LabPicsChemistry.zip + LabPicsMedical.zip).
# Annotation format: per-image folder with Vessels/ + Materials/ subdirs
# and a JSON annotation describing each instance. Verified once
# LabPics data lands; keys below are likely values from V2 spec.
# Class names from Categories.json (verified on disk).
# Pick the most-specific subclass when an annotation lists multiple (the
# generic "Vessel" appears in every annotation; we skip it and use the
# second class if available — see manifest builder).
VECTOR_LABPICS_VESSEL_MAP = {
    "Vessel":               ("vessel",                "Container"),  # fallback only
    "Syringe":              ("syringe",               "Container"),
    "Pippete":              ("pipette",               "Instrument"),  # sic - dataset typo
    "Tube":                 ("test tube",             "Container"),
    "IVBag":                ("IV bag",                "Container"),
    "DripChamber":          ("drip chamber",          "Container"),
    "IVBottle":             ("IV bottle",             "Container"),
    "Beaker":               ("beaker",                "Container"),
    "RoundFlask":           ("round-bottom flask",    "Container"),
    "Cylinder":             ("graduated cylinder",    "Container"),
    "SeparatoryFunnel":     ("separating funnel",     "Container"),
    "Funnel":               ("funnel",                "Container"),
    "Burete":               ("burette",               "Container"),  # sic - dataset typo
    "ChromatographyColumn": ("chromatography column", "Instrument"),
    "Condenser":            ("condenser",             "Instrument"),
    "Bottle":               ("bottle",                "Container"),
    "Jar":                  ("jar",                   "Container"),
    "Connector":            ("connector",             "Instrument"),
    "Flask":                ("flask",                 "Container"),
    "Cup":                  ("cup",                   "Container"),
    "Bowl":                 ("bowl",                  "Container"),
    "Erlenmeyer":           ("Erlenmeyer flask",      "Container"),
    "Vial":                 ("vial",                  "Container"),
    "Dish":                 ("petri dish",            "Container"),
    "HeatingVessel":        ("heating vessel",        "Container"),
    # Lab tools (class IDs 251+)
    "MagneticStirer":       ("magnetic stirrer",      "Instrument"),
    "Thermometer":          ("thermometer",           "Instrument"),
    "Spatula":              ("spatula",               "Instrument"),
    "Holder":                ("clamp holder",         "Instrument"),
    "Filter":               ("filter",                "Instrument"),
    "PipeTubeStraw":        ("tubing",                "Instrument"),
}

VECTOR_LABPICS_MATERIAL_MAP = {
    "Liquid":               ("liquid",                "Material"),
    "Foam":                 ("foam",                  "Material"),
    "Suspension":           ("suspension",            "Material"),
    "Solid":                ("solid material",        "Material"),
    "Filled":               ("liquid",                "Material"),   # "Filled" ≈ container has liquid
    "Powder":               ("powder",                "Material"),
    "Gel":                  ("gel",                   "Material"),
    "Granular":             ("granular solid",        "Material"),
    "SolidLargChunk":       ("solid chunk",           "Material"),
    "Vapor":                ("vapor",                 "Material"),
    "Other Material":       ("unknown material",      "Material"),
    "Urine":                ("urine",                 "Material"),
    "Blood":                ("blood",                 "Material"),
}


# ============================================================
# Physics-27 — 27 physics classes from data.yaml
# ============================================================
#
# Source: figshare 30984658 ("Physics Laboratory Equipment Image Dataset").
# 5,121 imgs (3586 train / 1023 valid / 512 test), YOLO labels.
# CC BY 4.0. data.yaml `names:` list is verbatim below.
#
# Entity type rules:
#   - measurement-emitting tools (multimeter, stopwatch, scale) → Measurement
#   - everything else → Instrument
PHYSICS27_MAP = {
    "Analog AC Ammeter":              ("analog AC ammeter",         "Measurement"),
    "Bar pendulum":                   ("bar pendulum",              "Instrument"),
    "Breadboard":                     ("breadboard",                "Instrument"),
    "Dc power Supply":                ("DC power supply",           "Instrument"),
    "Deflection Magnetometer":        ("deflection magnetometer",   "Measurement"),
    "Digital Multimeter":             ("digital multimeter",        "Measurement"),
    "Digital Stopwatch":              ("digital stopwatch",         "Measurement"),
    "Falling Plate":                  ("falling plate",             "Instrument"),
    "Hydrogen-Deuterium lamp":        ("hydrogen-deuterium lamp",   "Instrument"),
    "Lens":                           ("lens",                      "Instrument"),
    "Magnifying Glass":               ("magnifying glass",          "Instrument"),
    "Micrometer Screw Gauge":         ("micrometer screw gauge",    "Measurement"),
    "Newton-s Ring Microscope":       ("Newton's ring microscope",  "Instrument"),
    "Nylon Hammer":                   ("nylon hammer",              "Instrument"),
    "Physical Pendulum":              ("physical pendulum",         "Instrument"),
    "Polarimeter":                    ("polarimeter",               "Instrument"),
    "Power Supply":                   ("power supply",              "Instrument"),
    "Retort Stand":                   ("retort stand",              "Instrument"),
    "Scale":                          ("scale",                     "Measurement"),
    "Slotted Mass Set with Hanger":   ("slotted mass set",          "Instrument"),
    "Sodium Vapour Lamp Transformer": ("sodium vapour lamp transformer", "Instrument"),
    "Spherometer":                    ("spherometer",               "Measurement"),
    "Spring":                         ("spring",                    "Instrument"),
    "Stewart and Gee-s apparatus":    ("Stewart and Gee's apparatus", "Instrument"),
    "Vernier Caliper":                ("vernier caliper",           "Measurement"),
    "Weight":                         ("weight",                    "Instrument"),
    "Weight With Box":                ("weight with box",           "Instrument"),
}


# ============================================================
# Wikimedia Commons targeted crawl — 26 high-priority categories
# ============================================================
#
# Source: Wikimedia Commons API crawl with 1-level subcategory recursion.
# Captured via scripts/v8_crawl_wikimedia.py. Per-image attribution
# (license, author, source_url) stored in <slug>/manifest.jsonl.
#
# Key = category slug (lowercase, underscores).
WIKIMEDIA_MAP = {
    # Imaging / microscopy
    "atomic_force_microscopes":          ("atomic force microscope",          "Instrument"),
    "scanning_electron_microscope":      ("scanning electron microscope",     "Instrument"),
    "transmission_electron_microscopes": ("transmission electron microscope", "Instrument"),
    "stereo_microscopes":                ("optical microscope",               "Instrument"),
    "confocal_microscopes":              ("confocal microscope",              "Instrument"),
    # Physics measurement
    "oscilloscopes":                     ("oscilloscope",                     "Instrument"),
    "mass_spectrometers":                ("mass spectrometer",                "Instrument"),
    "nmr_spectrometer":                  ("NMR spectrometer",                 "Instrument"),
    "infrared_spectrometers":            ("infrared spectrometer",            "Instrument"),
    "spectrophotometers":                ("UV-Vis spectrometer",              "Instrument"),
    # Optics / lasers
    "optical_tables":                    ("optical table",                    "Instrument"),
    "lasers":                            ("laser",                            "Instrument"),
    # Bioengineering / nanofab
    "microfluidic_chips":                ("microfluidic device",              "Instrument"),
    "vacuum_chambers":                   ("vacuum chamber",                   "Instrument"),
    "sputter_coating":                   ("sputter deposition system",        "Instrument"),
    "chemical_vapour_deposition":        ("chemical vapor deposition system", "Instrument"),
    "photolithography":                  ("photolithography aligner",         "Instrument"),
    # Biology / medicine
    "gel_electrophoresis":               ("gel electrophoresis apparatus",    "Instrument"),
    "centrifuges":                       ("centrifuge",                       "Instrument"),
    "thermocyclers":                     ("PCR thermocycler",                 "Instrument"),
    "incubators":                        ("laboratory incubator",             "Instrument"),
    "surgical_instruments":              ("surgical instrument",              "Instrument"),
    # General laboratory containers
    "round-bottom_flasks":               ("round-bottom flask",               "Container"),
    "erlenmeyer_flasks":                 ("Erlenmeyer flask",                 "Container"),
    "beaker":                            ("beaker",                           "Container"),
    "petri_dishes":                      ("Petri dish",                       "Container"),
}


# ============================================================
# Helpers
# ============================================================

def chemeq25_label_to_unified(raw_label: str) -> tuple[str, str] | None:
    """Return (label, entity_type) for ChemEq25 class string, or None."""
    return CHEMEQ25_MAP.get(raw_label.strip())


def vector_labpics_label_to_unified(raw_label: str,
                                              category: str) -> tuple[str, str] | None:
    """`category` is "vessel" or "material" — picks the right map."""
    raw_label = (raw_label or "").strip()
    if category == "vessel":
        return VECTOR_LABPICS_VESSEL_MAP.get(raw_label)
    if category == "material":
        return VECTOR_LABPICS_MATERIAL_MAP.get(raw_label)
    return None


def physics27_label_to_unified(raw_label: str) -> tuple[str, str] | None:
    """Return (label, entity_type) for Physics-27 class string, or None."""
    return PHYSICS27_MAP.get(raw_label.strip())


def wikimedia_slug_to_unified(slug: str) -> tuple[str, str] | None:
    """Return (label, entity_type) for a Wikimedia category slug, or None.

    Slugs are produced by `scripts/v8_crawl_wikimedia.py::slugify`.
    """
    return WIKIMEDIA_MAP.get((slug or "").strip().lower())


def all_label_maps() -> dict[str, dict]:
    """For logging / introspection."""
    return {
        "chemeq25": CHEMEQ25_MAP,
        "vector_labpics_vessels": VECTOR_LABPICS_VESSEL_MAP,
        "vector_labpics_materials": VECTOR_LABPICS_MATERIAL_MAP,
        "physics27": PHYSICS27_MAP,
        "wikimedia": WIKIMEDIA_MAP,
    }
