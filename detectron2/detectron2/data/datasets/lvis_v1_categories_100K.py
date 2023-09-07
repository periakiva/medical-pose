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
LVIS_CATEGORIES = [{'name': 'hand (right)', 'instance_count': 34902, 'def': 'fruit with red or yellow or green skin and sweet to tart crisp whitish flesh', 'synonyms': ['hand (right)'], 'image_count': 1207, 'id': 1, 'frequency': 'f', 'synset': 'apple.n.01'}, {'name': 'hand (left)', 'instance_count': 34902, 'def': 'fruit with red or yellow or green skin and sweet to tart crisp whitish flesh', 'synonyms': ['hand (left)'], 'image_count': 1207, 'id': 2, 'frequency': 'f', 'synset': 'apple.n.01'}]