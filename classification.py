from gensim.models import Word2Vec
import pickle
import json
import sys
import editdistance
from time import time
import re, string;

from poleval.lib.poleval import strip_string, add_to_map, get_pickled, save_to_file, get_test_data, \
    get_polish_stopwords, get_label, EntityTuple


def get_mapping_classes(jsons, types):
    _category_map = {}
    _tuples = []
    _pl_map = {}
    _en_map = {}
    _disambiguation = {}
    _disambiguation_helper = {}
    for _json in jsons:
        root_type = ''
        if 'P31' in _json.keys():
            for subtype in _json['P31']:
                if subtype in types:
                    _category_map[_json['id']] = subtype
                    root_type = subtype
        elif 'P279' in _json.keys():
            for subtype in _json['P279']:
                if subtype in types:
                    _category_map[_json['id']] = subtype
                    root_type = subtype

        if _json['wiki']['pl'] and len(_json['wiki']['pl']) > 1:
            __cleaned_entity = strip_string(_json['wiki']['pl'])
            _tuple = EntityTuple(_json['labels']['en'], _json['labels']['pl'], _json['id'], root_type,
                                 _json['wiki']['pl'], __cleaned_entity)
            _tuples.append(_tuple)
            add_to_map(_pl_map, _disambiguation, _disambiguation_helper, _json['wiki']['pl'], _tuple, root_type)
    return _category_map, _tuples, _pl_map, _en_map, _disambiguation


# def get_inv_mapping(_category_mappings, _type_jsons):
#     inv_map = {}
#     for key, value in _category_mappings.items():


def process_batches(_i):
    print("Now processing test_tuples_chunk_{}".format(_i))
    _test_tuples = get_pickled("chunks/test_tuples_chunk_{}".format(_i))
    _ply = len(_test_tuples)
    print("#test_tuples_chunk: {}".format(_ply))
    t = time()
    _found, _not_found, _errors = validate(_test_tuples, entity_tuples, pl_map, prefix_map, lemma_map, _ply)
    print('Time to process test_tuples-{} : {} mins'.format(_i, round((time() - t) / 60, 2)))
    save_to_file("results_chunks-{}".format(_i), (_found, _not_found, _errors))


def process_test_tuples(_i):
    print("Now processing file: {}".format(saved_data_file.format(_i)))
    _data = get_pickled(saved_data_file.format(_i))
    _test_tuples = get_test_data(_data)
    save_to_file("test_tuples-{}".format(_i), _test_tuples)


def validate(_list_dict, _entity_tuples, _pl_map, _prefix_map, _lemma_map, _ply):
    _found = []
    _not_found = []
    _errors = []
    cnt = 0
    _current = set()
    _stopwords = get_polish_stopwords()
    for _dict in _list_dict:
        cnt += 1
        print("Progress {}".format(cnt)) if cnt % int(_ply / 10) == 0 else False
        for _label, _words in _dict.items():
            joined = ' '.join(map(lambda x: strip_string(x).lower(), _words))
            joined2 = ''.join(map(lambda x: strip_string(x).lower(), _words))
            if joined2 + _label not in _current:
                _current.add(joined2 + _label)
                _t = get_label(joined, _entity_tuples, _pl_map, _prefix_map, _lemma_map, _stopwords)
                if _t and len(_t) == 1:
                    _t = _t[0]
                    if _t and _t.original_type_id == _label:
                        _found.append((_label, _words, _t))
                    elif _t:
                        _errors.append((_label, _words, _t))
                elif _t and len(_t) > 1:
                    if len(list(filter(lambda x: x.original_type_id == _label, _t))) > 0:
                        _found.append(list(filter(lambda x: x.original_type_id == _label, _t)))
                    elif len(list(filter(lambda x: x.original_type_id == _label, _t))) == 0:
                        _errors.append((_label, _words, _t))
                else:
                    _not_found.append((_label, _words))

    return _found, _not_found, _errors


def validate_debug(_list_tuples, _entity_tuples, _pl_map, _prefix_map, _lemma_map, _ply):
    _found = []
    _not_found = []
    _errors = []
    cnt = 0
    for _label, _words, __x in _list_tuples:
        joined = ' '.join(_words)
        _t = get_label(joined, _entity_tuples, _pl_map, _prefix_map, _lemma_map)
        if _t and _t.original_type_id == _label:
            _found.append(_t)
        elif _t:
            _errors.append((_label, _words, _t))
        else:
            _not_found.append((_label, _words))
    return _found, _not_found, _errors


# entity_types_file_output = 'entity-types'
# type_jsons = get_pickled(entity_types_file_output)
# category_map, entity_tuples, pl_map, en_map, disambiguation = get_mapping_classes(type_jsons, categories_dict)
# entity_tuples.sort(key=lambda x: x.cleaned_original_entity)
# prefix_map = get_prefix_map(entity_tuples)
# save_to_file("mapping-objects", (category_map, entity_tuples, pl_map, en_map, disambiguation, prefix_map))


saved_data_file = "test_tuples-{}"
# data = get_pickled(saved_data_file.format(5))
#
# test_tuples = get_test_data(data)
# save_to_file("test_tuples", test_tuples)
# test_tuples = get_pickled("test_tuples")

# _found, _not_found, _errors = get_pickled("results")

# w2vec_model = Word2Vec.load(dir + "all-sentences-word2vec-m3.model")

lemma_map = get_pickled("lemma_map_ext")

(category_map, entity_tuples, pl_map, en_map, disambiguation, prefix_map) = get_pickled("mapping-objects_ext")
_t1 = get_label("rzymie", entity_tuples, pl_map, prefix_map, lemma_map,
                get_polish_stopwords())
# _t1 = get_label("system operacyjny", entity_tuples, pl_map, prefix_map, lemma_map,
#                 get_polish_stopwords())
# _t1 = get_label("system operacyjny systemów operacyjnych", entity_tuples, pl_map, prefix_map, lemma_map,
#                 get_polish_stopwords())
#
# _t1 = get_label("asocjacyjny", entity_tuples, pl_map, prefix_map, lemma_map,
#                 get_polish_stopwords())
# _t2 = get_label("energię", entity_tuples, pl_map, prefix_map, lemma_map)
# _found, _not_found, _errors = get_pickled("errors")
# validate_debug(_errors, entity_tuples, pl_map, prefix_map, lemma_map, 1000)

# multi = int(sys.argv[1])
multi = 95
process_batches(multi)

