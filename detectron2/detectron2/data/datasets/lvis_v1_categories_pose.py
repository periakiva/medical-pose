# Copyright (c) Facebook, Inc. and its affiliates.
# Autogen with
# with open("lvis_v1_val.json", "r") as f:
#     a = json.load(f)
# c = a["categories"]
# for x in c:
#     del x["image_count"]
#     del x["instance_count"]
# LVIS_CATEGORIES = repr(c) + "  # noqa"
# with open("/tmp/lvis_categories.py", "wt") as f:
#     f.write(f"LVIS_CATEGORIES = {LVIS_CATEGORIES}")
# Then paste the contents of that file below

# fmt: off
# LVIS_CATEGORIES = [{'id': 1, 'name': 'casualty', 'instances_count': 2561, 'def': '', 'synonyms': ['casualty'], 'image_count': 2448, 'frequency': '', 'synset': ''}, {'id': 2, 'name': 'user', 'instances_count': 1017, 'def': '', 'synonyms': ['user'], 'image_count': 1012, 'frequency': '', 'synset': ''}]
LVIS_CATEGORIES = [{'id': 1, 'name': 'patient', 'instances_count': 5972, 'def': '', 'synonyms': ['patient'], 'image_count': 5972, 'frequency': 'f', 'synset': ''}, {'id': 2, 'name': 'user', 'instances_count': 5935, 'def': '', 'synonyms': ['user'], 'image_count': 5935, 'frequency': 'f', 'synset': ''}]