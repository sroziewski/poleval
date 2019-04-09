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


categories_dict = {'Q5': "człowiek", 'Q2221906': "położenie geograficzne", 'Q11862829': "dyscyplina naukowa",
                   'Q4936952': "struktura anatomiczna", 'Q12737077': "zajęcie", 'Q29048322': "model pojazdu",
                   'Q811430': "konstrukcja",
                   'Q47461344': "utwór pisany", 'Q6999': "ciało niebieskie", 'Q11460': "odzież", 'Q16521': "takson",
                   'Q24334685': "byt mityczny", 'Q31629': "dyscyplina sportu",
                   'Q28855038': "istota nadprzyrodzona", 'Q11435': "ciecz", 'Q28108': "system polityczny",
                   'Q16334298': "zwierzę", 'Q43460564': "substancja chemiczna", 'Q732577': "publikacja",
                   'Q271669': "ukształtowanie terenu", 'Q34770': "język", 'Q2198779': "jednostka",
                   'Q20719696': "obiekt geograficzny", 'Q15621286': "dzieło artystyczne", 'Q39546': "narzędzie",
                   'Q7239': "organizm", 'Q2095': "jedzenie"}

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


i = 1
