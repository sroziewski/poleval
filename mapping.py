import json
import pickle

from poleval.lib.poleval import get_soup, get_divs, get_features, filter_out_blinds, get_pickled, recursion_limit, \
    save_to_file


def to_terminate(_original_id, _type_id, lp31, lp279, lp17, _breakable):
    _level = 20
    if _type_id == 'P31' and lp31 > _level:
        _breakable.append(True)
        print("{} {}".format(_type_id, _original_id))
        return True
    if _type_id == 'P279' and lp279 > _level:
        _breakable.append(True)
        print("{} {}".format(_type_id, _original_id))
        return True
    if _type_id == 'P17' and lp17 > _level:
        _breakable.append(True)
        print("{} {}".format(_type_id, _original_id))
        return True
    return False


def get_ids(_id, _type_id, _categories_dict, _global_map, _original_id, _breakable, _recursion_level_p31,
            _recursion_level_p279, _recursion_level_p17):
    if to_terminate(_original_id, _type_id, _recursion_level_p31, _recursion_level_p279, _recursion_level_p17,
                    _breakable):
        return
    if len(_breakable) > 0:
        return
    features = filter_out_blinds(get_features(get_divs(get_soup(_id), _type_id)))
    found_mapping = extract_existing_mapping(features, _categories_dict)
    if len(found_mapping) > 0:
        _global_map[_original_id] = found_mapping[0]
        _breakable.append(True)
        return

    for feat in features:
        if feat not in _categories_dict and len(_breakable) == 0:
            get_ids(feat, "P279", _categories_dict, _global_map, _original_id, _breakable, _recursion_level_p31,
                    _recursion_level_p279 + 1, _recursion_level_p17)
            get_ids(feat, "P31", _categories_dict, _global_map, _original_id, _breakable, _recursion_level_p31 + 1,
                    _recursion_level_p279, _recursion_level_p17)
            get_ids(feat, "P17", categories_dict, _global_map, _original_id, _breakable, _recursion_level_p31,
                    _recursion_level_p279, _recursion_level_p17 + 1)


def get_mapping(_id, _global_map):
    original_id = _id
    breakable = []
    get_ids(original_id, "P279", categories_dict, _global_map, original_id, breakable, 0, 0, 0)
    if len(breakable) == 0:
        breakable = []
        try:
            get_ids(original_id, "P31", categories_dict, _global_map, original_id, breakable, 0, 0, 0)
            if len(breakable) == 0:
                breakable = []
                get_ids(original_id, "P17", categories_dict, _global_map, original_id, breakable, 0, 0, 0)
        except RuntimeError:
            print(original_id)


def extract_main_entity_category2(jsons, types, _category_vectors, _model_v2w, _categories, categories_dict):
    category_map = {}
    _outsiders = []
    for _json in jsons:
        flag = 0
        if 'P31' in _json.keys():
            for subtype in _json['P31']:
                if subtype in types:
                    category_map[_json['id']] = subtype
                    flag = 1
        elif 'P279' in _json.keys():
            for subtype in _json['P279']:
                if subtype in types:
                    category_map[_json['id']] = subtype
                    flag = 1
        if flag == 0:
            _outsiders.append(_json)
    return _outsiders


input_file = dir + 'tokens-with-entities-and-tags_1mln.tsv'
saved_data_file = "20000/tokens-with-entities-and-tags_1mln"
json_file = dir + 'entities.jsonl'
entity_types_file = dir + 'entity-types.tsv'
entity_types_file_output = 'entity-types'

categories_dict = {'Q5': "człowiek", 'Q2221906': "położenie geograficzne", 'Q11862829': "dyscyplina naukowa",
                   'Q4936952': "struktura anatomiczna", 'Q12737077': "zajęcie", 'Q29048322': "model pojazdu",
                   'Q811430': "konstrukcja",
                   'Q47461344': "utwór pisany", 'Q6999': "ciało niebieskie", 'Q11460': "odzież", 'Q16521': "takson",
                   'Q24334685': "byt mityczny", 'Q31629': "dyscyplina sportu",
                   'Q28855038': "istota nadprzyrodzona", 'Q11435': "ciecz", 'Q28108': "system polityczny",
                   'Q16334298': "zwierzę", 'Q43460564': "substancja chemiczna", 'Q732577': "publikacja",
                   'Q271669': "ukształtowanie terenu", 'Q34770': "język", 'Q2198779': "jednostka",
                   'Q20719696': "obiekt geograficzny", 'Q15621286': "dzieło artystyczne", 'Q39546': "narzędzie",
                   'Q7239': "organizm", 'Q2095': "jedzenie", 'Q7184903': "obiekt abstrakcyjny", 'Q483247': "zjawisko",
                   'Q11344': "substancja", 'Q6671777': "struktura"}

categories = ["człowiek", "położenie geograficzny", "dyscyplina naukowy", "struktura anatomiczny", "zajęcie",
              "model pojazd", "konstrukcja", "utwór pisany", "ciało niebieskie", "odzież", "takson", "byt mityczny",
              "dyscyplina sport", "istota nadprzyrodzony", "ciecz", "system polityczny", "zwierzę",
              "substancja chemiczny", "publikacja", "ukształtowanie teren", "język", "jednostka", "obiekt geograficzny",
              "dzieło artystyczny",
              "narzędzie", "organizm", "jedzenie"]

# entity_types = get_entity_types(entity_types_file)
# / type_jsons = read_json_file(json_file)
# / save_to_file(entity_types_file_output, type_jsons)
# type_jsons = get_pickled(entity_types_file_output)
#
#
# a = w2vec_model_3.wv.cosine_similarities((w2vec_model_3.wv.get_vector("kraj")+w2vec_model_3.wv.get_vector("federacja")+w2vec_model_3.wv.get_vector("niepodległy")+w2vec_model_3.wv.get_vector("kolonialny"))/4,  category_vectors)
#
# outsiders = get_outsiders(type_jsons, entity_types)
# save_to_file("outsiders", outsiders)

outsiders = get_pickled("outsiders")

global_map = {}
# get_mapping("Q23165129", global_map)

multi = 2

with recursion_limit(1500):
    global_map = {}
    cnt = 100000 * (multi-1)
    for _id in outsiders[(multi-1)*100000:multi*100000]:
        get_mapping(_id, global_map)
        cnt += 1
        print("Progress {}".format(cnt)) if cnt % 10000 == 0 else False

save_to_file("global_map.{}".format(multi), global_map)
