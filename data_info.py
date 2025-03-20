LITHOLOGY_MAPPING = {
    1: "BASALT",
    2: "BRECCIA",
    3: "CLAY-SILT-SAND-GRAVEL",
    4: "LIMESTONE",
    5: "LIMESTONE-MUDSTONE",
    6: "MUDSTONE",
    7: "MUDSTONE-HALITE",
    8: "MUDSTONE-LIMESTONE",
    9: "MUDSTONE-SANDSTONE",
    10: "SAND-CLAY-GRAVEL",
    11: "SAND-GRAVEL",
    12: "SANDSTONE",
    13: "SANDSTONE-BRECCIA",
    14: "GRAVEL",
    15: "SAND"
}

SUBSTRATE_MAPPING = {
    1: "BEDROCK",
    2: "VENEER",
    3: "SUPERFICIAL"
}

LANDUSE_MAPPING = {
    1: "Deciduous woodland",
    2: "Coniferous woodland",
    3: "Arable",
    4: "Improved grassland",
    5: "Neutral grassland",
    6: "Calcareous grassland",
    7: "Acid grassland",
    8: "Fen",
    9: "Heather",
    10: "Heather grassland",
    11: "Bog",
    12: "Inland rock",
    13: "Saltwater",
    14: "Freshwater",
    15: "Supralittoral rock",
    16: "Supralittoral sediment",
    17: "Littoral rock",
    18: "Littoral sediment",
    19: "Saltmarsh",
    20: "Urban",
    21: "Suburban"
}

BUILDING_TYPE_MAPPING = {
    1: "Buildings",
    2: "Land",
    3: "Rail",
    4: "Roads",
    5: "Structures",
    6: "Water"
}

CONTINUOUS = ['ELEVATION', 'SLOPE', 'IMPERVIOUSNESS', 'NDVI', 'DISTANCE TO RIVER', 'DISTANCE TO ROAD']
CATEGORICAL = ['LANDUSE','LITHOLOGY','SUBSTRATE','BUILDING TYPE']
MAPPINGS = [LANDUSE_MAPPING, LITHOLOGY_MAPPING, SUBSTRATE_MAPPING, BUILDING_TYPE_MAPPING]