import json

import editdistance
import pickle
import re
import requests
from bs4 import BeautifulSoup
from gensim.models import Word2Vec
import numpy as np

dir = '/home/szymon/juno/challenge/poleval/'


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


class DataItem(object):
    def __init__(self, array):
        self.doc_id = array[0]
        self.token = array[1]
        self.lemma = array[2].strip()
        self.preceding_space = array[3]
        self.morphosyntactic_tags = array[4]
        self.link_title = array[5]
        self.entity_id = array[6]


class Page(object):
    def __init__(self, array):
        self.page_id = array[0]
        self.title = array[1]
        self.type = array[2]


class ArticleParent(object):
    def __init__(self, array):
        self.article_id = array[0]
        self.categories = array[1:]


class CategoryParent(object):
    def __init__(self, array):
        self.category_id = array[0]
        self.parent_categories = array[1:]


class ChildArticle(object):
    def __init__(self, array):
        self.category_id = array[0]
        self.articles = array[1:]


class ChildCategory(object):
    def __init__(self, array):
        self.category_id = array[0]
        self.child_categories = array[1:]


class LinkBySource(object):
    def __init__(self, array):
        self.source_id = array[0]
        self.article_names = array[1:]


class WordTuple(object):
    def __init__(self, lemma, type):
        self.lemma = lemma
        self.type = type


def get_pickled(filename):
    with open(dir + filename + '.pickle', 'rb') as handle:
        data = pickle.load(handle)
        handle.close()
        return data


def cleaning(doc_list, stopwords):
    # Lemmatizes and removes stopwords
    # doc needs to be a spacy Doc object
    txt = [token for token in doc_list if not token.word in stopwords]
    # Word2Vec uses context words to learn the vector representation of a target word,
    # if a sentence is only one or two words long,
    # the benefit for the training is very small
    if len(txt) > 0:
        return txt


class Word(object):
    def __init__(self, word, pos, entity_id):
        self.word = word
        self.pos = pos
        self.entity_id = entity_id


class WordTriplet(object):
    def __init__(self, lemma, morpho_type, entity_id):
        self.lemma = lemma
        self.moprho_type = morpho_type
        self.entity_id = entity_id


class Entity(object):
    def __init__(self, entity, context):
        self.entity = entity
        self.context = context

    def add_context(self, _c):
        self.context.extend(_c)


def get_word_tuples(docs):
    tuples = []
    for doc in docs:
        value = doc.lemma
        if len(doc.link_title) > len(doc.lemma):
            value = doc.link_title
        _id = doc.entity_id if doc.entity_id != '_' else ''
        tuples.append(WordTriplet(value, doc.morphosyntactic_tags, _id))
    return tuples


def get_polish_stopwords():
    with open(dir + "polish.stopwords.txt") as file:
        return [line.strip() for line in file]


def get_clean_text(token_lists, stopwords):
    return [x for x in [cleaning(tokens, stopwords) for tokens in token_lists] if x is not None]


def strip_string(_entity):
    if _entity:
        pattern = re.compile('[\W_]+')
        return pattern.sub('', _entity).lower()


def strip_string_space(_entity):
    pattern1 = re.compile('[\s]+')
    pattern2 = re.compile('[\W_]+')
    pattern3 = re.compile('XQX')
    return pattern3.sub(' ', pattern2.sub('', pattern1.sub('XQX', _entity))).lower()


def get_sentences(tuples):
    sentences = []
    handy = []
    for _ in tuples:
        if _.moprho_type == 'interp:' and _.lemma == '.':
            sentences.append(handy)
            handy = []
        elif _.moprho_type != 'interp:':
            handy.append(Word(strip_string(_.lemma.split(':')[0]), _.moprho_type, _.entity_id))
    if handy:
        sentences.append(handy)
    return sentences


def contains_entity(_words):
    _ent = []
    for _word in _words:
        if _word.entity_id:
            _ent.append(_word)
    return _ent if len(_ent) > 0 else False


def save_to_file(filename, obj):
    with open(dir + filename + '.pickle', 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.close()


def extract_context(_sentence, _entity, __context):
    _l = []
    for _word in _sentence:
        if _word.pos.split(":")[0] == _entity.pos.split(":")[0]:
            _l.append(_word) if not _word.entity_id == _entity.entity_id and _word.word not in __context else None
    return _l


def flatten_list(_list):
    return [item for sublist in _list for item in sublist]


def add_to_list(_sentence_list, _sentences):
    for _s in _sentences:
        _sentence_list.append(_s)


def list_docs_to_sentences(docs):
    _sentence_list = []
    stopwords = get_polish_stopwords()
    for key, doc_list in docs.items():
        _sentences = get_clean_text(get_sentences(get_word_tuples(doc_list)), stopwords)
        if len(_sentences) > 0:
            add_to_list(_sentence_list, _sentences)
    return _sentence_list


def get_test_data(data):
    _tuples = []
    for _data_item_list in data.values():
        _test_tuples = {}
        for _data_item in _data_item_list:
            if _data_item.entity_id != '_':
                if _data_item.entity_id not in _test_tuples:
                    _test_tuples[_data_item.entity_id] = [strip_string(_data_item.token)]
                elif _data_item.token not in _test_tuples[_data_item.entity_id]:
                    _test_tuples[_data_item.entity_id].append(strip_string(_data_item.token))
        if len(_test_tuples) > 0:
            _tuples.append(_test_tuples)
    return _tuples


def manage_doc_context(_sentence_list):
    _was_entity = False
    _c = 0
    _entity_map = {}
    for _sentence in _sentence_list:
        if _was_entity:
            for _e in _was_entity:
                try:
                    _l2 = extract_context(_sentence_list[_c], _e, _entity_map[_e.entity_id].context)
                    _entity_map[_e.entity_id].add_context(_l2)
                except:
                    i = 1
        _entity_l = contains_entity(_sentence)
        if _entity_l:
            _l1 = False
            _m1 = {}
            for _e in _entity_l:
                if _c - 1 >= len(_sentence_list):
                    _l1 = extract_context(_sentence_list[_c - 1], _e, [])
                _l0 = extract_context(_sentence_list[_c], _e, [])
                if _l1:
                    _l1.extend(_l0)
                    _l = _l1
                else:
                    _l = _l0
                _m1[_e] = _l
            _was_entity = _entity_l
            for _e in _entity_l:
                if _e.entity_id not in _entity_map:
                    _entity_map[_e.entity_id] = Entity(_e, _m1[_e])
        else:
            _was_entity = False
        _c += 1
    return _entity_map


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


def merge_maps(_map1, _map2):
    return {**_map1, **_map2}


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


def get_soup(_pl_entity_wiki):
    r = requests.get("https://www.wikidata.org/wiki/" + _pl_entity_wiki)
    return BeautifulSoup(r._content, 'html.parser')


def get_entity(_e, _lemma_map, _stopwords):
    _t = strip_string(_e).lower()
    if _t and _t not in _stopwords:
        return _lemma_map[_t] if not _e.isdigit() and _t in _lemma_map else [_t]


def extract_main_entity_category(jsons, types, _category_vectors, _model_v2w, _categories, categories_dict):
    category_map = {}
    outsiders = []
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
            outsiders.append(_json)



def read_json_file(json_file):
    with open(json_file) as f:
        _data = [json.loads(line) for line in f]
    f.close()
    return _data


def get_entity_types(filename):
    with open(filename) as f:
        types = {line.split('/')[-1].strip(): line.split('/')[-1].strip() for line in f}
    f.close()
    return types


def map_docs_to_sentences(docs):
    data_map = {}
    stopwords = get_polish_stopwords()
    for key, doc_list in docs.items():
        txt = get_clean_text(get_sentences(get_word_tuples(doc_list)), stopwords)
        if len(txt) > 0:
            data_map[key] = txt
    return data_map


def get_prefix_map(_entity_tuples):
    _prefix_map = {}
    cnt = 0
    for _e in _entity_tuples:
        if len(_e.cleaned_original_entity) > 0 and not _e.cleaned_original_entity[0] in _prefix_map:
            _prefix_map[_e.cleaned_original_entity[0]] = cnt
        cnt += 1
    return _prefix_map


def get_label(_entity, _entity_tuples, _pl_map, _prefix_map, _lemma_map, _stopwords):
    if _entity:
        try:
            _entity = strip_dangling_keywords(_entity)
        except IndexError:
            i = 1
    else:
        return
    if ''.join(_entity.split()) in pl_map:
        __f = _pl_map[''.join(_entity.split())]
        if len(_entity.split()) == len(__f.cleaned_original_entity_space.split()):
            return [__f]
    _lemma_entities = []
    for _e in _entity.split():
        _lemmas = get_entity(_e, _lemma_map, _stopwords)
        try:
            if _lemmas: _lemma_entities.append(_lemmas)
        except AttributeError:
            i = 1
    if len(_lemma_entities) == 1:
        for _lemma in _lemma_entities[0]:
            return find_tuple(_entity_tuples, [_lemma], _lemma_map, _pl_map, _prefix_map)
    if len(_lemma_entities) == 2:
        _i_size = len(_lemma_entities[0])
        _j_size = len(_lemma_entities[1])
        for _i in range(0, _i_size):
            for _j in range(0, _j_size):
                _l = [_lemma_entities[0][_i], _lemma_entities[1][_j]]
                return find_tuple(_entity_tuples, _l, _lemma_map, _pl_map, _prefix_map)
    if len(_lemma_entities) == 3:
        _i_size = len(_lemma_entities[0])
        _j_size = len(_lemma_entities[1])
        _k_size = len(_lemma_entities[2])
        for _i in range(0, _i_size):
            for _j in range(0, _j_size):
                for _k in range(0, _k_size):
                    _l = [_lemma_entities[0][_i], _lemma_entities[1][_j], _lemma_entities[2][_k]]
                    return find_tuple(_entity_tuples, _l, _lemma_map, _pl_map, _prefix_map)
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
                        return find_tuple(_entity_tuples, _le, _lemma_map, _pl_map, _prefix_map)
    if len(_lemma_entities) > 4:
        try:
            return find_tuple(_entity_tuples, list(map(lambda x: x[0], _lemma_entities)), _lemma_map, _pl_map,
                              _prefix_map)
        except IndexError:
            i = 1


def find_by_entity(_entity, _mapping_tuples, _pl_map, _prefix_map, _lemma_map, _entities):
    r = []
    if _entity in _pl_map:
        __f = _pl_map[_entity]
        if len(_entity.split()) == len(__f.cleaned_original_entity_space.split()):
            return [__f]
    try:
        _index = _prefix_map[_entity[0]] if _entity[0] in _prefix_map else _prefix_map['z']
        for _tuple in _mapping_tuples[_index:_index + get_upper_bound(
                _mapping_tuples[_index], _mapping_tuples[_index:])]:
            try:
                if _tuple.similar_to(_entity, _entities):
                    r.append(_tuple)
            except AttributeError:
                i = 1
    except:
        j = 1
    return r


def get_upper_bound(_lower_tuple, _tuples):
    _upper = 0
    _lower = ''
    if _lower_tuple.cleaned_original_entity:
        _lower = _lower_tuple.cleaned_original_entity[0]
    for _tuple in _tuples:
        if _tuple.cleaned_original_entity:
            if _lower != _tuple.cleaned_original_entity[0]:
                return _upper
        _upper += 1
    return _upper


def find_tuple(_entity_tuples, _l, _lemma_map, _pl_map, _prefix_map):
    if len(_l) == 1:
        _f = find_by_entity(_l[0], _entity_tuples, _pl_map, _prefix_map, _lemma_map, _l)
        if _f: return _f
    if len(_l) == 2:
        _f = find_by_entity(_l[0] + _l[1], _entity_tuples, _pl_map, _prefix_map, _lemma_map, _l)
        if _f: return _f
        _f = find_by_entity(_l[1] + _l[0], _entity_tuples, _pl_map, _prefix_map, _lemma_map, _l)
        if _f: return _f
    if len(_l) == 3:
        _f = find_by_entity(_l[0] + _l[1] + _l[2], _entity_tuples, _pl_map, _prefix_map, _lemma_map, _l)
        if _f: return _f
        _f = find_by_entity(_l[0] + _l[2] + _l[1], _entity_tuples, _pl_map, _prefix_map, _lemma_map, _l)
        if _f: return _f
        _f = find_by_entity(_l[2] + _l[1] + _l[0], _entity_tuples, _pl_map, _prefix_map, _lemma_map, _l)
        if _f: return _f
        _f = find_by_entity(_l[2] + _l[0] + _l[1], _entity_tuples, _pl_map, _prefix_map, _lemma_map, _l)
        if _f: return _f
        _f = find_by_entity(_l[1] + _l[2] + _l[0], _entity_tuples, _pl_map, _prefix_map, _lemma_map, _l)
        if _f: return _f
        _f = find_by_entity(_l[1] + _l[0] + _l[2], _entity_tuples, _pl_map, _prefix_map, _lemma_map, _l)
        if _f: return _f
    if len(_l) > 3:
        j = ''.join(_l)
        _f = find_by_entity(j, _entity_tuples, _pl_map, _prefix_map, _lemma_map, _l)
        if _f: return _f
    return None


def get_test_data(data):
    _tuples = []
    for _data_item_list in data.values():
        _test_tuples = {}
        for _data_item in _data_item_list:
            if _data_item.entity_id != '_':
                if _data_item.entity_id not in _test_tuples:
                    _test_tuples[_data_item.entity_id] = [strip_string(_data_item.token)]
                elif _data_item.token not in _test_tuples[_data_item.entity_id]:
                    _test_tuples[_data_item.entity_id].append(strip_string(_data_item.token))
        if len(_test_tuples) > 0:
            _tuples.append(_test_tuples)
    return _tuples


def chunks(l, _n, _i):
    _step = int(len(l) / _n)
    if _i + 1 == _n:
        return l[_i * _step:]
    else:
        return l[_i * _step:(_i + 1) * _step]


def get_mentions(docs):
    return '+'.join(map(lambda x: x.link_title, filter(lambda d: len(d.link_title) > 1, docs)))


class Mention(object):
    def __init__(self, sentence, with_entity):
        self.sentence = sentence
        self.with_entity = with_entity


def get_sentences_with_mentions(triplets):
    sentences = []
    handy = []
    is_entity = False

    for _ in triplets:
        if _.type == 'interp' and _.lemma == '.':
            sentences.append(Mention(handy, is_entity))
            is_entity = False
            handy = []
        elif _.type != 'interp':
            if _.is_entity:
                is_entity = True
            handy.append(_.lemma.split(':')[0].lower().translate(
                {ord(i): None for i in ['”', '„', '(', ')', '{', '}', '[', ']']}))
    if handy:
        sentences.append(Mention(handy, is_entity))
    return sentences


def add_to_map(__map, __disambiguation, __disambiguation_helper, __entity, __tuple, __root_type):
    if not __entity:
        return
    __entity = strip_string(__entity)
    if __entity in __map.keys():
        if __entity not in __disambiguation:
            __disambiguation[__entity] = [__disambiguation_helper[__entity]]
            del (__disambiguation_helper[__entity])
        if __tuple not in __disambiguation[__entity]:
            __disambiguation[__entity].append(__tuple)
    else:
        __map[__entity] = __tuple
        __disambiguation_helper[__entity] = __tuple