# coding=utf-8
from gensim.models import Word2Vec
import pickle
import json
import editdistance
from time import time
import re, string;

dir = '/home/szymon/juno/challenge/poleval/'


def strip_string(_entity):
    pattern = re.compile('[\W_]+')
    return pattern.sub('', _entity)


def strip_string_space(_entity):
    to_exclude = ['”', '„', '(', ')', '{', '}', '[', ']', '.', ',', '!', '?', '%', '\'', '\\', '/', '+', '-', '=',
                  ':', '*', '@', '#', '$', '^', '&', '_', '<', '>', '"', '÷', '’']
    _entity = _entity.lower().translate(
        {ord(i): None for i in to_exclude}) if _entity else False
    return _entity


def levenshtein(s, t):
    return editdistance.eval(s, t)


class EntityTuple(object):
    def __init__(self, entity_en, entity_pl, original_type_id, root_type_id, pl_wiki, cleaned_pl_wiki):
        self.original_entity = pl_wiki
        self.cleaned_original_entity_space = strip_string_space(pl_wiki)
        self.original_type_id = original_type_id
        self.cleaned_original_entity = cleaned_pl_wiki
        self.root_type_id = root_type_id
        self.entity_en = strip_string(entity_en)
        self.entity_pl = strip_string(entity_pl)
        self.set_lev_sensitivity()
        self.set_lev_similarity()

    def set_lev_similarity(self):
        if self.cleaned_original_entity:
            if len(self.cleaned_original_entity) <= 5:
                self.lev_similarity = 3
            if len(self.cleaned_original_entity) > 5:
                self.lev_similarity = 4

    def set_lev_sensitivity(self):
        if self.cleaned_original_entity:
            if len(self.cleaned_original_entity) <= 3:
                self.lev_sensitivity = 0
            if len(self.cleaned_original_entity) > 3 and len(self.cleaned_original_entity) < 6:
                self.lev_sensitivity = 2
            if len(self.cleaned_original_entity) >= 6 and len(self.cleaned_original_entity) < 10:
                self.lev_sensitivity = 3
            if len(self.cleaned_original_entity) >= 10 and len(self.cleaned_original_entity) < 14:
                self.lev_sensitivity = 4
            if len(self.cleaned_original_entity) >= 14 and len(self.cleaned_original_entity) < 18:
                self.lev_sensitivity = 5
            if len(self.cleaned_original_entity) >= 18 and len(self.cleaned_original_entity) < 22:
                self.lev_sensitivity = 6
            if len(self.cleaned_original_entity) >= 22 and len(self.cleaned_original_entity) < 26:
                self.lev_sensitivity = 7
            if len(self.cleaned_original_entity) >= 26 and len(self.cleaned_original_entity) < 30:
                self.lev_sensitivity = 8
            if len(self.cleaned_original_entity) >= 30 and len(self.cleaned_original_entity) < 34:
                self.lev_sensitivity = 9
            if len(self.cleaned_original_entity) >= 34:
                self.lev_sensitivity = 10

        elif self.entity_en:
            if len(self.entity_en) > 3 and len(self.entity_en) < 5:
                self.lev_sensitivity = 1
            if len(self.entity_en) >= 5 and len(self.entity_en) < 7:
                self.lev_sensitivity = 2
            if len(self.entity_en) >= 7 and len(self.entity_en) < 11:
                self.lev_sensitivity = 3
            if len(self.entity_en) >= 11 and len(self.entity_en) < 14:
                self.lev_sensitivity = 4
            if len(self.entity_en) >= 14 and len(self.entity_en) < 18:
                self.lev_sensitivity = 5
            if len(self.entity_en) >= 18 and len(self.entity_en) < 22:
                self.lev_sensitivity = 6
            if len(self.entity_en) >= 22 and len(self.entity_en) < 26:
                self.lev_sensitivity = 7
            if len(self.entity_en) >= 26 and len(self.entity_en) < 30:
                self.lev_sensitivity = 8
            if len(self.entity_en) >= 30 and len(self.entity_en) < 34:
                self.lev_sensitivity = 9
            if len(self.entity_en) >= 34:
                self.lev_sensitivity = 10

    def similar_to(self, _entity, _entities):
        # to_exclude = [' ', '”', '„', '(', ')', '{', '}', '[', ']', '.', ',']
        # _entity = _entity.lower().translate(
        #     {ord(i): None for i in to_exclude}) if _entity else False
        _l = self.cleaned_original_entity_space.split()
        if len(_l) > 2 and len(_entities) > 2 and _l[0][:self.lev_similarity] == _entities[0][:self.lev_similarity] and \
                _l[2][:self.lev_similarity] == _entities[2][:self.lev_similarity] and _l[1][
                                                                                      :self.lev_similarity] == \
                _entities[1][:self.lev_similarity] and self.lev_sensitivity and levenshtein(
            self.cleaned_original_entity, _entity) < self.lev_sensitivity:
            return True
        if len(_l) > 1 and len(_entities) > 1 and _l[0][:self.lev_similarity] == _entities[0][:self.lev_similarity] and \
                _l[1][
                :self.lev_similarity] == \
                _entities[1][:self.lev_similarity] and self.lev_sensitivity and levenshtein(
            self.cleaned_original_entity, _entity) < self.lev_sensitivity:
            return True

        if _l[0][:self.lev_similarity] == _entities[0][:self.lev_similarity] and self.lev_sensitivity and levenshtein(
                self.cleaned_original_entity, _entity) < self.lev_sensitivity:
            return True


def get_pickled(filename):
    with open(dir + filename + '.pickle', 'rb') as handle:
        data = pickle.load(handle)
        handle.close()
        return data


def merge_maps(_map1, _map2):
    return {**_map1, **_map2}


def flatten_list(list):
    return [item for sublist in list for item in sublist]


def get_polish_stopwords():
    with open(dir + "polish.stopwords.txt") as file:
        return [line.strip() for line in file]


def get_sensitivity(_entity):
    if len(_entity) <= 3:
        return 0
    if len(_entity) > 3 and len(_entity) < 6:
        return 2
    if len(_entity) >= 6 and len(_entity) < 10:
        return 3
    if len(_entity) >= 10 and len(_entity) < 14:
        return 4
    if len(_entity) >= 14 and len(_entity) < 18:
        return 5
    if len(_entity) >= 18 and len(_entity) < 22:
        return 6
    if len(_entity) >= 22 and len(_entity) < 26:
        return 7
    if len(_entity) >= 26 and len(_entity) < 30:
        return 8
    if len(_entity) >= 30 and len(_entity) < 34:
        return 9
    if len(_entity) >= 34:
        return 10


def contains(_r, _e):
    for __e in _r:
        if levenshtein(__e, _e) <= min(get_sensitivity(_e),
                                       get_sensitivity(__e)):
            return True
    return False


def put_label_for_key(_key, _merged, _label, _l):
    # try:
    not_found_list_of_tuples.append(EntityTuple('', _key, _label, '', '', ' '.join(_l)))
    # except TypeError:
    #     i = 1
    if _key not in _merged:
        _merged[_key] = [_label]
    elif _label not in _merged[_key]:
        _merged[_key].append(_label)


def then_append(_r, _e, _checked):
    for __e in _r:
        if levenshtein(__e, _e) > min(get_sensitivity(_e),
                                      get_sensitivity(__e)) and not contains(_r, _e):
            _r.append(_e)
            return


def strip_dangling_keywords(_in):
    # _in = _entity.split()
    try:
        _r = [_in[0]]
    except IndexError:
        i = 1
    _checked = set()
    for _e in _in:
        then_append(_r, _e, _checked)
    return _r


def key_by_entity(_l, _merged, _label):
    # try:
    if len(_l) == 1:
        put_label_for_key(_l[0], _merged, _label, _l)
    if len(_l) == 2:
        put_label_for_key(_l[0] + _l[1], _merged, _label, _l)
        put_label_for_key(_l[1] + _l[0], _merged, _label, _l)
    if len(_l) == 3:
        put_label_for_key(_l[0] + _l[1] + _l[2], _merged, _label, _l)
        put_label_for_key(_l[0] + _l[2] + _l[1], _merged, _label, _l)
        put_label_for_key(_l[2] + _l[1] + _l[0], _merged, _label, _l)
        put_label_for_key(_l[2] + _l[0] + _l[1], _merged, _label, _l)
        put_label_for_key(_l[1] + _l[2] + _l[0], _merged, _label, _l)
        put_label_for_key(_l[1] + _l[0] + _l[2], _merged, _label, _l)
    if len(_l) == 4:
        put_label_for_key(_l[0] + _l[1] + _l[2] + _l[3], _merged, _label, _l)
        put_label_for_key(_l[0] + _l[2] + _l[1] + _l[3], _merged, _label, _l)
        put_label_for_key(_l[2] + _l[1] + _l[0] + _l[3], _merged, _label, _l)
        put_label_for_key(_l[2] + _l[0] + _l[1] + _l[3], _merged, _label, _l)
        put_label_for_key(_l[1] + _l[2] + _l[0] + _l[3], _merged, _label, _l)
        put_label_for_key(_l[1] + _l[0] + _l[2] + _l[3], _merged, _label, _l)
        put_label_for_key(_l[0] + _l[1] + _l[3] + _l[2], _merged, _label, _l)
        put_label_for_key(_l[0] + _l[2] + _l[3] + _l[1], _merged, _label, _l)
        put_label_for_key(_l[2] + _l[1] + _l[3] + _l[0], _merged, _label, _l)
        put_label_for_key(_l[2] + _l[0] + _l[3] + _l[1], _merged, _label, _l)
        put_label_for_key(_l[1] + _l[2] + _l[3] + _l[0], _merged, _label, _l)
        put_label_for_key(_l[1] + _l[0] + _l[3] + _l[2], _merged, _label, _l)
    if len(_l) > 4:
        j = ''.join(map(lambda x: x[0], _l))
        put_label_for_key(j, _merged, _label, list(map(lambda x: x[0], _l)))
    # except TypeError:
    #     i = 1


def save_to_file(filename, obj):
    with open(dir + filename + '.pickle', 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.close()


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


def clean_tuples(_list):
    _new_list2 = []
    for _tuple in _list:
        _e1 = extract_tuple_text(_tuple[1])
        _e1.extend(extract_tuple_text(_tuple[2])) if _tuple[2] else None
        _new_list2.append((_tuple[0], set(flatten_list(_e1))))
    return _new_list2


def get_entity(_e):
    _t = strip_string(_e).lower()
    if _t and _t not in stopwords:
        return lemma_map[_t] if not _e.isdigit() and _t in lemma_map else [_t]


def extract_tuple_text(_text):
    _l1 = []
    for _t in _text.split():
        if _t:
            _ee = get_entity(_t, lemma_map, stopwords)
            if _ee:
                _l1.append(_ee)
    return _l1


def get_entity(_e, _lemma_map, _stopwords):
    _t = strip_string(_e).lower()
    if _t and _t not in _stopwords:
        return _lemma_map[_t] if not _e.isdigit() and _t in _lemma_map else [_t]


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

# merged = merge_maps(global_map1, global_map2)


i = 1
