# coding=utf-8

import requests
from bs4 import BeautifulSoup
import sys
from gensim.models import Word2Vec
import pickle
import editdistance
from time import time
import re, string;

dir = '/home/szymon/juno/challenge/poleval/'


def get_soup(_pl_entity_wiki):
    r = requests.get("https://www.wikidata.org/wiki/" + _pl_entity_wiki)
    return BeautifulSoup(r._content, 'html.parser')


def get_pickled(filename):
    with open(dir + filename + '.pickle', 'rb') as handle:
        data = pickle.load(handle)
        handle.close()
        return data


def get_divs(_soup, type_id):
    r = _soup.findAll("div", {"id": type_id})
    if r:
        return r[0]
    else:
        return None


def get_text(divs):
    text = ''
    if not divs:
        return []
    content = get_divs(divs, "bodyContent")
    if content:
        ps = content.findAll("p")
        for p in ps[:4]:
            text += p.getText()

    return text


def save_to_file(filename, obj):
    with open(dir + filename + '.pickle', 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.close()


def get_subclasses(divs_subclasses):
    subclasses = []
    for div_instance in divs_subclasses:
        subclasses.append(div_instance.findAll("a")[0].attrs['title'])
    return subclasses


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


def get_entity(_e, _lemma_map, _stopwords):
    _t = strip_string(_e).lower()
    if _t and _t not in _stopwords:
        return _lemma_map[_t] if not _e.isdigit() and _t in _lemma_map else [_t]


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


def put_label_for_key(_key, _merged, _label, _l):
    # try:
    # not_found_list_of_tuples.append(EntityTuple('', _key, _label, '', '', ' '.join(_l)))
    # except TypeError:
    #     i = 1
    if _key not in _merged:
        _merged[_key] = [_label]
    elif _label not in _merged[_key]:
        _merged[_key].append(_label)


def key_by_entity(_l, _merged, _true_label, _false_labels):
    # try:
    if len(_l) == 1:
        put_label_for_key(_l[0], _merged, _true_label, _l)
    if len(_l) == 2:
        put_label_for_key(_l[0] + _l[1], _merged, _true_label, _l)
        put_label_for_key(_l[1] + _l[0], _merged, _true_label, _l)
    if len(_l) == 3:
        put_label_for_key(_l[0] + _l[1] + _l[2], _merged, _true_label, _l)
        put_label_for_key(_l[0] + _l[2] + _l[1], _merged, _true_label, _l)
        put_label_for_key(_l[2] + _l[1] + _l[0], _merged, _true_label, _l)
        put_label_for_key(_l[2] + _l[0] + _l[1], _merged, _true_label, _l)
        put_label_for_key(_l[1] + _l[2] + _l[0], _merged, _true_label, _l)
        put_label_for_key(_l[1] + _l[0] + _l[2], _merged, _true_label, _l)
    if len(_l) == 4:
        put_label_for_key(_l[0] + _l[1] + _l[2] + _l[3], _merged, _true_label, _l)
        put_label_for_key(_l[0] + _l[2] + _l[1] + _l[3], _merged, _true_label, _l)
        put_label_for_key(_l[2] + _l[1] + _l[0] + _l[3], _merged, _true_label, _l)
        put_label_for_key(_l[2] + _l[0] + _l[1] + _l[3], _merged, _true_label, _l)
        put_label_for_key(_l[1] + _l[2] + _l[0] + _l[3], _merged, _true_label, _l)
        put_label_for_key(_l[1] + _l[0] + _l[2] + _l[3], _merged, _true_label, _l)
        put_label_for_key(_l[0] + _l[1] + _l[3] + _l[2], _merged, _true_label, _l)
        put_label_for_key(_l[0] + _l[2] + _l[3] + _l[1], _merged, _true_label, _l)
        put_label_for_key(_l[2] + _l[1] + _l[3] + _l[0], _merged, _true_label, _l)
        put_label_for_key(_l[2] + _l[0] + _l[3] + _l[1], _merged, _true_label, _l)
        put_label_for_key(_l[1] + _l[2] + _l[3] + _l[0], _merged, _true_label, _l)
        put_label_for_key(_l[1] + _l[0] + _l[3] + _l[2], _merged, _true_label, _l)
    if len(_l) > 4:
        j = ''.join(map(lambda x: x[0], _l))
        put_label_for_key(j, _merged, _true_label, list(map(lambda x: x[0], _l)))


class Word(object):
    def __init__(self, word, pos, entity_id):
        self.word = word
        self.pos = pos
        self.entity_id = entity_id


class Entity(object):
    def __init__(self, entity, context):
        self.entity = entity
        self.context = context

    def add_context(self, _c):
        self.context.extend(_c)


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


i = 1

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


i = 1
