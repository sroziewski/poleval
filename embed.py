import editdistance
import pickle
import re
from gensim.models import Word2Vec
import numpy as np

from poleval.lib.poleval import levenshtein, flatten_list, strip_string, get_entity, get_polish_stopwords, get_pickled, \
    save_to_file

input_file = dir + 'test/task3_test.tsv'
saved_data_file = "test/task3_test"


# data_object_map(input_file, saved_data_file)
# data = get_pickled(saved_data_file)
# test_tuples = get_test_data(data)


# saved_data_file = "20000/tokens-with-entities-and-tags_1mln"

# data = get_pickled(saved_data_file)
# sentences = list_docs_to_sentences(data)
# save_to_file("sentence_list", sentences)
# manage_doc_context(list_docs_to_sentences(data))
# sentences = get_pickled("sentence_list")
# entity_map = manage_doc_context(sentences)
# save_to_file("entity_map-valid", entity_map)


def get_avg_similarity(_tuple, _context, _w2vec):
    _avg_l = []
    for _w1 in _context:
        for _w2 in _tuple[1]:
            try:
                _avg_l.append(_w2vec.wv.distance(_w1, _w2))
            except KeyError:
                pass
    return np.mean(_avg_l)


def filter_singleton_tuple(_key, _tuples):
    _n_t = []
    for __t in _tuples:
        if len(__t[1]) == 1 and list(__t[1])[0] == _key:
            pass
        else:
            _n_t.append(__t)
    return _n_t


def map_to_valid(_entity_valid_map, _found, _found_with_error, _wikidata_context_map, _w2vec, _scrap_found_map):
    _mapped_test = []
    _not_mapped = []
    _hash_map = {}
    for _k, _v in _entity_valid_map.items():
        if _v.entity.word in _wikidata_context_map:
            _tuples = _wikidata_context_map[_v.entity.word]
            _max_l = []
            _context = set(list(map(lambda x: x.word, _v.context)))
            __clean_tuples = filter_singleton_tuple(_v.entity.word, _tuples)
            if len(__clean_tuples) == 0:
                __strip_Q_in_tuple_name = []
                for _x in _tuples:
                    __strip_Q_in_tuple_name.append(int(_x[0][1:]))
                __strip_Q_found = 'Q{}'.format(min(__strip_Q_in_tuple_name))
                _mapped_test.append((_v.entity.entity_id, __strip_Q_found))
                _hash_map[_v.entity.word] = __strip_Q_found
            else:
                for _t in __clean_tuples:
                    _max_l.append((_t, get_avg_similarity(_t, _context, _w2vec)))

                _found_tuple = min(_max_l, key=lambda _item: _item[1])
                _mapped_test.append((_v.entity.entity_id, _found_tuple[0][0]))
                _hash_map[_v.entity.word] = _found_tuple[0][0]
            i = 1
        elif _v.entity.word in _found:
            _mapped_test.append((_v.entity.entity_id, _found[_v.entity.word][0]))
            _hash_map[_v.entity.word] = _found[_v.entity.word][0]
        elif _v.entity.word in _found_with_error:
            _mapped_test.append((_v.entity.entity_id, _found_with_error[_v.entity.word][0]))
            _hash_map[_v.entity.word] = _found_with_error[_v.entity.word][0]
        else:
            if _v.entity.entity_id in _scrap_found_map:
                _mapped_test.append((_v.entity.entity_id, _scrap_found_map[_v.entity.entity_id]))
                _hash_map[_v.entity.word] = _scrap_found_map[_v.entity.entity_id]
            else:
                if _v.entity.word in _hash_map:
                    _mapped_test.append((_v.entity.entity_id, _hash_map[_v.entity.word]))
                else:
                    _mapped_test.append((_v.entity.entity_id, ''))
                    _not_mapped.append((_k, _v))

    return _mapped_test, _not_mapped


def filter_tuples(_word, _clean_tuples_):
    _filtered = []
    for _t_ in _clean_tuples_:
        _has = False
        for _w_ in _t_[1]:
            if _w_[0:3] == _word[0:3]:
                _has = True
        if _has:
            _filtered.append(_t_)
    return _filtered


def filter_found(_tuples):
    _filtered_found = []
    for ___t in _tuples:
        if len(___t[0][0]) < 7:
            _filtered_found.append(___t)
    return _filtered_found


def map_scrapped_not_found(_tuples, _w2vec):
    _scrapped_mapped_test = []
    _scrapped_not_mapped_test = []

    for _t in _tuples:
        if _t[0] == 'e1753':
            jj = 1
        _max_l = []
        _context = set(list(map(lambda x: x.word, _t[1].context)))
        if len(_t[2]) == 0:
            _scrapped_not_mapped_test.append((_t[1].entity.entity_id, _t[1].entity.word))
        else:
            __clean_tuples = filter_singleton_tuple(_t[1].entity.word, _t[2])
            __filtered_tuples = filter_tuples(_t[1].entity.word, __clean_tuples)
            if len(__filtered_tuples) == 0:
                __strip_Q_in_tuple_name = []
                for _x in _t[2]:
                    __strip_Q_in_tuple_name.append(int(_x[0][1:]))
                __strip_Q_found = 'Q{}'.format(min(__strip_Q_in_tuple_name))
                _scrapped_mapped_test.append((_t[1].entity.entity_id, __strip_Q_found))
            else:
                for __t in __filtered_tuples:
                    _max_l.append((__t, get_avg_similarity(__t, _context, _w2vec)))

                _max_l_f = filter_found(_max_l)
                if len(_max_l_f) > 0:
                    _found_tuple = min(_max_l_f, key=lambda _item: _item[1])
                else:
                    _found_tuple = min(_max_l, key=lambda _item: _item[1])
                _scrapped_mapped_test.append((_t[1].entity.entity_id, _found_tuple[0][0]))

    return _scrapped_mapped_test, _scrapped_not_mapped_test


def filter_found_with_error(_found_with_error):
    _filtered = {}
    for _k, _v in _found_with_error.items():
        if len(_v) == 1:
            _filtered[_k] = _v
    return _filtered


def filter_found_with_error_more(_found_with_error):
    _filtered = {}
    for _k, _v in _found_with_error.items():
        if len(_v) > 1:
            _filtered[_k] = _v
    return _filtered


def try_to_find(_not_found_errors_list_of_tuples_chunk, _entity):
    _i = 0
    _candidates = {}
    for _et in _not_found_errors_list_of_tuples_chunk:
        try:
            if _et.original_type_id not in _candidates and levenshtein(_et.cleaned_original_entity, _entity) < _et.lev_sensitivity - 1:
                _candidates[_et.original_type_id] = _et
                if len(_candidates) > 3:
                    return _candidates
        except:
            AttributeError
        _i += 1
    return _candidates


def clean_tuples(_list):
    _new_list2 = []
    for _tuple in _list:
        _e1 = extract_tuple_text(_tuple[1])
        _e1.extend(extract_tuple_text(_tuple[2])) if _tuple[2] else None
        _new_list2.append((_tuple[0], set(flatten_list(_e1))))
    return _new_list2


def extract_tuple_text(_text):
    _l1 = []
    for _t in _text.split():
        if _t:
            _ee = get_entity(_t)
            if _ee:
                _l1.append(_ee)
    return _l1


def map_scrap_found(_scrap_found_and_not_found):
    _scrap_found_map = {}
    for _t in _scrap_found_and_not_found:
        _scrap_found_map[_t[0]] = _t[1]
    return _scrap_found_map


stopwords = get_polish_stopwords()
lemma_map = get_pickled("lemma_map_ext")

found_with_error = get_pickled("merged_map_found_with_error")
found = get_pickled("merged_map_found")
# (category_map, entity_tuples, pl_map, en_map, disambiguation, prefix_map) = get_pickled("mapping-objects_ext")
wikidata_context_map = get_pickled("wikidata_context_map")
entity_valid_map = get_pickled("entity_map-valid")

scrap_found_map = get_pickled("scrap_found_map")

filtered_found_with_error = filter_found_with_error(found_with_error)
filtered_found_with_error_more = filter_found_with_error_more(found_with_error)

w2vec = Word2Vec.load(dir + "all-sentences-word2vec-m3.model")

# mapped, not_mapped = map_to_valid(entity_valid_map, found, filtered_found_with_error, wikidata_context_map, w2vec, scrap_found_map)
# save_to_file("map_test_set_with_scrap_filled", (mapped, not_mapped))

(mapped, not_mapped) = get_pickled("map_test_set_with_scrap_filled")

import csv
with open(dir+'roziewski-poleval-task3.tsv', 'w') as tsvfile:
    writer = csv.writer(tsvfile, delimiter='=')
    for _t in mapped:
        writer.writerow([_t[0], _t[1]])

(mapped, not_mapped) = get_pickled("map_test_set_with_scrap")
# (mapped, not_mapped) = get_pickled("map_test_set")


clean_scrapped_not_found = get_pickled("clean_scrapped_not_found")
scrap_mapped, scrap_not_mapped = get_pickled("scrap_found_and_not_found")

scrap_found_map = map_scrap_found(scrap_mapped)
save_to_file("scrap_found_map", scrap_found_map)

# scrap_mapped, scrap_not_mapped = map_scrapped_not_found(clean_scrapped_not_found, w2vec)
# save_to_file("scrap_found_and_not_found", (scrap_mapped, scrap_not_mapped))

i = 1
# clean_scrapped_not_found = []
# for _t in scrapped_not_found:
#     _clean_tuple = (_t[0], _t[1], clean_tuples(_t[2]))
#     clean_scrapped_not_found.append(_clean_tuple)
#
# save_to_file("clean_scrapped_not_found", clean_scrapped_not_found)

# not_found_errors_list_of_tuples_chunk1 = get_pickled("not_mapped_found_candidates-5")

# _to_scrap = []
# for _t in not_found_errors_list_of_tuples_chunk1:
#     _to_scrap.append((_t[0][0], _t[0][1], list(_t[1].keys())))


# not_found_errors_list_of_tuples = get_pickled("not_found_errors_list_of_tuples")


# nf = get_pickled("not_mapped_found_candidates-2")
#
#
# scrapped_not_found = get_pickled("scrapped_not_found-{}".format(1))
# for i in range(2, 8):
#     _scrapped_not_found = get_pickled("scrapped_not_found-{}".format(i))
#     scrapped_not_found.extend(_scrapped_not_found)

multi = 1

# not_mapped_found_candidates = []
#
# k = 0
# for _t in not_mapped[0:50]:
#     print("script: {} , {}".format(multi, k))
#     _f = try_to_find(not_found_errors_list_of_tuples_chunk1, _t[1].entity.word)
#     for __k, __v in _f.items():
#         print("{} : {}".format(__k, _t[1].entity.word))
#     not_mapped_found_candidates.append((_t, _f))
#     k += 1
#
# save_to_file("not_mapped_found_candidates-{}".format(multi), not_mapped_found_candidates)
#
#
# f = try_to_find(not_found_errors_list_of_tuples_chunk1, 'wałęsa')


i = 1
