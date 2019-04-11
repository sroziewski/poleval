from poleval.lib.entity.definitions import categories_dict
from poleval.lib.poleval import get_ids


def get_mapping(_id, _global_map):
    original_id = _id
    breakable = []
    get_ids(original_id, "P31", categories_dict, _global_map, original_id, breakable)
    if len(breakable) == 0:
        breakable = []
        get_ids(original_id, "P279", categories_dict, _global_map, original_id, breakable)


# div1 = get_divs(get_soup("Q177"), "P31")
# div2 = get_divs(get_soup("Q177"), "P279")
#
# f1 = get_features(div1)

global_map = {}


get_mapping("Q31", global_map)

# original_id = "Q31"


# get_ids(original_id, "P31", categories_dict, global_map, original_id, breakable)
# if len(breakable)==0:
#     breakable = []
#     get_ids(original_id, "P279", categories_dict, global_map, original_id, breakable)

# for f in f1:
#     if f not in categories_dict:
#         c = get_ids(f, "P31")
#         i = 1


# divs = soup.findAll("div", {"class":"wikibase-statementgrouplistview"})
# v = "P279" in r._content


# divsInstances = div1.findAll("div", {"class": "wikibase-snakview-variation-valuesnak"})
#
# divsSubclasses = div2.findAll("div", {"class": "wikibase-snakview-variation-valuesnak"})
#
# ins = get_instances(divsInstances)
# subs = get_instances(divsSubclasses)

# v1 = str(r._content).find("P279")
