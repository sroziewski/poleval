# coding=utf-8

from gensim.models import Word2Vec
import pickle
import json
import editdistance
from time import time
import re, string;

dir = '/home/szymon/juno/challenge/poleval/'


def get_pickled(filename):
    with open(dir + filename + '.pickle', 'rb') as handle:
        data = pickle.load(handle)
        handle.close()
        return data


def contains(_r, _e):
    for __e in _r:
        if levenshtein(__e, _e) <= min(get_sensitivity(_e),
                                      get_sensitivity(__e)):
            return True
    return False


def save_to_file(filename, obj):
    with open(dir + filename + '.pickle', 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.close()


def merge_lemma_map(_map1, _map2):
    for _key, _val in _map1.items():
        if _key in _map2:
            for _val2 in _map2[_key]:
                if _val2 not in _map1[_key]:
                    _map1[_key].append(_val2)
    return _map1

map0 = get_pickled("lemma_map-{}".format(0))

for i in range(1, 19):
    _map = get_pickled("lemma_map-{}".format(i))
    map0 = merge_lemma_map(map0, _map)

save_to_file("lemma_map_ext", map0)