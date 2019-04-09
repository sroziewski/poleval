# coding=utf-8

from poleval.lib.poleval import strip_string, strip_dangling_keywords, key_by_entity, flatten_list, get_entity, \
    clean_tuples, get_polish_stopwords, get_pickled, save_to_file


def merge_lists(_list1, _lemma_map, _stopwords, _merged_map):
    for _label, _words in _list1:
        _words = strip_dangling_keywords(list(map(lambda x: strip_string(x).lower(), _words)))
        _lemma_entities = []
        for _e in _words:
            _entity_lemma = get_entity(_e, _lemma_map, _stopwords)
            if _entity_lemma: _lemma_entities.append(_entity_lemma)
        if len(_lemma_entities) == 1:
            for _lemma in _lemma_entities[0]:
                key_by_entity([_lemma], _merged_map, _label)
            # _key = ''.join(_lemma_entities[0])
        if len(_lemma_entities) == 2:
            _i_size = len(_lemma_entities[0])
            _j_size = len(_lemma_entities[1])
            for _i in range(0, _i_size):
                for _j in range(0, _j_size):
                    _l = [_lemma_entities[0][_i], _lemma_entities[1][_j]]
                    key_by_entity(_l, _merged_map, _label)
        if len(_lemma_entities) == 3:
            _i_size = len(_lemma_entities[0])
            _j_size = len(_lemma_entities[1])
            _k_size = len(_lemma_entities[2])
            for _i in range(0, _i_size):
                for _j in range(0, _j_size):
                    for _k in range(0, _k_size):
                        _l = [_lemma_entities[0][_i], _lemma_entities[1][_j], _lemma_entities[2][_k]]
                        key_by_entity(_l, _merged_map, _label)
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
                            key_by_entity(_le, _merged_map, _label)
        if len(_lemma_entities) > 4:
            key_by_entity(_lemma_entities, _merged_map, _label)


def merge_tuple_map(_map1, _map2):
    for _key, _list in _map2.items():
        if _key not in _map1:
            _map1[_key] = _list
        else:
            _map1[_key].extend(_list)


def remove_disamb_pages(_map):
    _clean_map = {}
    for _key, _list in _map.items():
        if _key == 'złoty':
            i = 1
        _new_list = []
        for _tuple in _list:
            if _tuple[1]:
                if 'ujednoznaczniająca' not in _tuple[2].split():
                    _tuple[2].split()
                    _new_list.append(_tuple)
        if len(_new_list) > 0:
            _clean_map[_key] = clean_tuples(_new_list)
    return _clean_map


stopwords = get_polish_stopwords()
lemma_map = get_pickled("lemma_map_ext")

tuple_map0 = remove_disamb_pages(get_pickled("tuple_map_scrap-{}".format(2)))
for i in range(3,37):
    if i != 6:
        tuple_map1 = remove_disamb_pages(get_pickled("tuple_map_scrap-{}".format(i)))
        merge_tuple_map(tuple_map0, tuple_map1)

save_to_file("wikidata_context_map", tuple_map0)

# lemma_map = get_pickled("lemma_map-{}".format(0))
merged_map = {}
not_found_list_of_tuples = []
# global_map1 = get_pickled("global_map.10")
# global_map2 = get_pickled("global_map.13")
# outsiders = get_pickled("outsiders")
_found, _not_found, _errors = get_pickled("chunks/results_chunks-23")
lemma_map = get_pickled("lemma_map_ext")
(category_map, entity_tuples, pl_map, en_map, disambiguation, prefix_map) = get_pickled("mapping-objects_ext")
# list(filter(lambda x:type(x[2]).__name__=='list', _found))
# _errors_mapped = list(map(lambda x: (x[0], x[1]), _errors))
# merge_lists(_errors_mapped, lemma_map, stopwords, merged_map)
# merge_lists(_not_found, lemma_map, stopwords, merged_map)

for i in range(1, 95):
    _found, _not_found, _errors = get_pickled("chunks/results_chunks-{}".format(i))
    # _errors_mapped = list(map(lambda x: (x[0], x[1]), _errors))
    _found_filtered = list(map(lambda x: (x[0], x[1]), filter(lambda x:type(x[2]).__name__=='EntityTuple', _found)))
    # _found_filtered = list(filter(lambda x:type(x[2]).__name__=='EntityTuple', _found))
    # merge_lists(_errors_mapped, lemma_map, stopwords, merged_map)
    merge_lists(_found_filtered, lemma_map, stopwords, merged_map)
    # merge_lists(_not_found, lemma_map, stopwords, merged_map)

# save_to_file("merged_map_not_found_errors", merged_map)
save_to_file("merged_map_found_with_error", merged_map)
save_to_file("found_with_error_list_of_tuples", not_found_list_of_tuples)
# save_to_file("not_found_errors_list_of_tuples", not_found_list_of_tuples)
