# coding=utf-8

import requests

from poleval.lib.poleval import strip_dangling_keywords, strip_string, get_entity, key_by_entity, get_pickled, \
    save_to_file


def process(tuples, _lemma_map, _stopwords):
    _merged_map = {}
    for _true_label, _words, _false_labels in tuples:
        _words = strip_dangling_keywords(list(map(lambda x: strip_string(x).lower(), _words)))
        _lemma_entities = []
        for _e in _words:
            _entity_lemma = get_entity(_e, _lemma_map, _stopwords)
            if _entity_lemma: _lemma_entities.append(_entity_lemma)
        if len(_lemma_entities) == 1:
            for _lemma in _lemma_entities[0]:
                key_by_entity([_lemma], _merged_map, _true_label, _false_labels)
            # _key = ''.join(_lemma_entities[0])
        if len(_lemma_entities) == 2:
            _i_size = len(_lemma_entities[0])
            _j_size = len(_lemma_entities[1])
            for _i in range(0, _i_size):
                for _j in range(0, _j_size):
                    _l = [_lemma_entities[0][_i], _lemma_entities[1][_j]]
                    key_by_entity(_l, _merged_map, _true_label, _false_labels)
        if len(_lemma_entities) == 3:
            _i_size = len(_lemma_entities[0])
            _j_size = len(_lemma_entities[1])
            _k_size = len(_lemma_entities[2])
            for _i in range(0, _i_size):
                for _j in range(0, _j_size):
                    for _k in range(0, _k_size):
                        _l = [_lemma_entities[0][_i], _lemma_entities[1][_j], _lemma_entities[2][_k]]
                        key_by_entity(_l, _merged_map, _true_label, _false_labels)
        if len(_lemma_entities) == 4:
            _i_size = len(_lemma_entities[0])
            _j_size = len(_lemma_entities[1])
            _k_size = len(_lemma_entities[2])
            _l_size = len(_lemma_entities[3])
            for _i in range(0, _i_size):
                for _j in range(0, _j_size):
                    for _k in range(0, _k_size):
                        for _li in range(0, _l_size):
                            _le = [_lemma_entities[0][_i], _lemma_entities[1][_j], _lemma_entities[2][_k],
                                   _lemma_entities[3][_li]]
                            key_by_entity(_le, _merged_map, _true_label, _false_labels)
        if len(_lemma_entities) > 4:
            key_by_entity(_lemma_entities, _merged_map, _true_label, _false_labels)


def get_json(_id):
    _url = 'https://www.wikidata.org/w/api.php?action=wbgetentities&ids={}&format=json'.format(_id)
    _resp = requests.get(url=_url)
    _data = _resp.json()
    _label = _description = ''
    if 'labels' in _data['entities'][_id]:
        if 'pl' in _data['entities'][_id]['labels']:
            _label = _data['entities'][_id]['labels']['pl']['value']
    if 'descriptions' in _data['entities'][_id]:
        if 'pl' in _data['entities'][_id]['descriptions']:
            _description = _data['entities'][_id]['descriptions']['pl']['value']
    return _id, _label, _description


def get_wikidata_words(_tuples):
    _tuple_map = {}
    _i = 0
    for _tuple in _tuples:
        _l = []
        for _id in _tuple[1]:
            _l.append(get_json(_id))
        _tuple_map[_tuple[0]] = _l
        _i += 1
    return _tuple_map


def scrap_not_found(_tuples):
    _scrapped_not_found = []
    for __t in _tuples:
        _l = []
        for _id in __t[2]:
            _l.append(get_json(_id))
        _scrapped_not_found.append((__t[0], __t[1], _l))


not_found_errors_list_of_tuples_chunk1 = get_pickled("not_mapped_found_candidates-{}".format(i))

to_scrap = []
for _t in not_found_errors_list_of_tuples_chunk1:
    to_scrap.append((_t[0][0], _t[0][1], list(_t[1].keys())))

scrapped_not_found = scrap_not_found(to_scrap)

save_to_file("scrapped_not_found-{}".format(i), scrapped_not_found)

# _found, _not_found, _errors = get_pickled("chunks/bin/results_chunks-83")
# lemma_map = get_pickled("lemma_map_ext")
# (category_map, entity_tuples, pl_map, en_map, disambiguation, prefix_map) = get_pickled("mapping-objects_ext")
# stopwords = get_polish_stopwords()

merged_map = get_pickled("merged_map_not_found_errors")
merged_map_filtered = list(filter(lambda x: len(x[1]) > 1, merged_map.items()))


multi = 10000
# multi = int(sys.argv[1])
j = 13

data = merged_map_filtered[(j-1)*multi:j*multi]

tuple_map = get_wikidata_words(data)
save_to_file("tuple_map_scrap-{}".format(j), tuple_map)

