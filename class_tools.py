CITYSCAPES_MAP = {
    "traffic sign": "signboard, sign",
    "terrain": "earth, ground",
    "rider": "person",
    "sidewalk": "road, route",
    "road": "road, route"
}
COCO_MAP = {
    "wall-other-merged": "wall",
    "building-other-merged": "building",
    "wall-brick": "wall",
    "wall-stone": "wall",
    "wall-tile": "wall",
    "wall-wood": "wall",
    "ceiling-merged": "ceiling",
    "window-other": "window",
    "window-blind": "window",
    "door-stuff": "door",
    "fence-merged": "fence",
    "stop sign": "signboard, sign",
    "sky-other-merged": "sky",
    "tree-merged": "tree",
    "grass-merged": "grass",
    "mountain-merged": "mountain, mount, hill",
    "rock-merged": "rock",
    "airplane": "plane",
    "couch": "sofa",
    "table-merged": "table",
    "dining table": "table",
    "cabinet-merged": "cabinet",
    "rug-merged": "rug",
    "mirror-stuff": "mirror",
    "backpack": "bag",
    "handbag": "bag",
    "bridge": "bridge, span",
    "refrigerator": "refrigerator, icebox",
    "paper-merged": "paper",
    "pavement-merged": "road, route",
    "floor-other-merged": "floor",
    "floor-wood": "floor",
    "food-other-merged": "food",
    "dirt-merged": "dirt",
    "blanket": "blanket, cover",
    "road": "road, route",
    "water-other": "water",
    "sports ball": "ball"
}
ADE20K_MAP = {
    "plant": "vegetation",
    "armchair": "chair",
    "swivel chair": "chair",
    "skyscraper": "building",
    "step, stair": "stairs",
    "stairway, staircase": "stairs",
    "palm, palm tree": "tree",
    "minibike, motorbike": "motorcycle",
    "van": "car",
    "pool table, billiard table, snooker table": "table",
    "coffee table": "table",
    "earth, ground": "land, ground, soil",
    "screen door, screen": "screen door",
    "crt screen": "screen",
    "sidewalk, pavement": "road, route",
    "toilet, can, commode, crapper, pot, potty, stool, throne": "can, commode, crapper, pot, potty, stool, throne",
    "dirt track": "road, route",
    "animal": "other animal",
    "kitchen island": "unknown",
    "cabinet": "chest of drawers, chest, bureau, dresser",
    "seat": "unknown",
    "hill": "mountain, mount, hill",
    "mountain, mount": "mountain, mount, hill",
    "washer, automatic washer, washing machine": "dishwasher"
}

global classes_list
with open("class_file/classes.txt", "r") as file:
    classes_list = file.read().splitlines()

global stuff_list
stuff_list = [
    "land, ground, soil",
    "rock, stone",
    "field",
    "path",
    "floor",
    "gravel",
    "dirt",
    "grass",
    "mountain, mount, hill",
    "ceiling",
    "snow",
    "water",
    "sea",
    "sky",
    "sand",
    "road, route",
    "river",
    "snow",
    "vegetation",
    "building",
    "column, pillar",
    "blind, screen",
    "window",
    "tree"
]

def converted_oneformer_label(class_semantic_label, dict_map):
    converted_label_list = []
    for semantic_label in class_semantic_label:
        converted_label = dict_map.get(semantic_label, semantic_label)
        converted_label_list.append(converted_label)
    return converted_label_list
