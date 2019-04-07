from gensim.models import Word2Vec
import pickle
import json
import sys
import editdistance
from time import time
import re, string;

dir = '/home/szymon/juno/challenge/poleval/'


def levenshtein(s, t):
    return editdistance.eval(s, t)


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


def get_mapping_classes2(jsons, types):
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
        if root_type != '':
            if _json['wiki']['pl'] and len(_json['wiki']['pl']) > 1:
                _tuple = EntityTuple(_json['labels']['en'], _json['labels']['pl'], _json['id'], root_type,
                                     _json['wiki']['pl'])
                _tuples.append(_tuple)
                # add_to_map(_pl_map, _disambiguation, _disambiguation_helper, _tuple.entity_pl, _tuple, root_type)
                add_to_map(_pl_map, _disambiguation, _disambiguation_helper, _json['wiki']['pl'], _tuple, root_type)
                # add_to_map(_en_map, _disambiguation, _disambiguation_helper, _json['wiki']['en'], _tuple, root_type)
                # add_to_map(_pl_map, _disambiguation, _tuple.entity_pl, _tuple, root_type)
                # if _tuple.entity_pl in _pl_map.keys():
                #     _key = _tuple.entity_pl if _tuple.entity_pl in _pl_map.keys() else False
                #     if not _key:
                #         i = 1
                #     if _key in _disambiguation:
                #         _disambiguation[_key].append(_tuple)
                #     else:
                #         _disambiguation[_key] = [_tuple]
                # elif _tuple.entity_en in _en_map.keys():
                #     _key = _tuple.entity_en if _tuple.entity_en in _en_map.keys() else False
                #     if not _key:
                #         i = 1
                #     if _key in _disambiguation:
                #         _disambiguation[_key].append(_tuple)
                #     else:
                #         _disambiguation[_key] = [_tuple]
                # else:
                #     _pl_map[_tuple.entity_pl] = root_type
                #     _en_map[_tuple.entity_en] = root_type
    return _category_map, _tuples, _pl_map, _en_map, _disambiguation


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


def get_pickled(filename):
    with open(dir + filename + '.pickle', 'rb') as handle:
        data = pickle.load(handle)
        handle.close()
        return data


def read_json_file(json_file):
    with open(json_file) as f:
        _data = [json.loads(line) for line in f]
    f.close()
    return _data


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
        if len(_l) != len(_entities):
            return False
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


def cleaning(doc_list, stopwords):
    # Lemmatizes and removes stopwords
    # doc needs to be a spacy Doc object
    txt = [token.strip() for token in doc_list if not token.strip() in stopwords]
    # Word2Vec uses context words to learn the vector representation of a target word,
    # if a sentence is only one or two words long,
    # the benefit for the training is very small
    if len(txt) > 2:
        return txt


class DataItem(object):
    def __init__(self, array):
        self.doc_id = array[0]
        self.token = array[1]
        self.lemma = array[2]
        self.preceding_space = array[3]
        self.morphosyntactic_tags = array[4]
        self.link_title = array[5]
        self.entity_id = array[6]


class WordTriplet(object):
    def __init__(self, lemma, type, labels):
        self.lemma = lemma
        self.type = type
        self.labels = labels

    def add_label(self, label):
        self.labels.append(label)


def get_clean_text(token_lists, stopwords):
    return [x for x in [cleaning(tokens, stopwords) for tokens in token_lists] if x is not None]


def get_polish_stopwords():
    with open(dir + "polish.stopwords.txt") as file:
        return [line.strip() for line in file]


def get_word_tuples(docs):
    tuples = []
    for doc in docs:
        value = doc.lemma
        if len(doc.link_title) > len(doc.lemma):
            value = doc.link_title
        tuples.append(WordTriplet(value, doc.morphosyntactic_tags, doc.entity_id != '_'))
    return tuples


def get_mentions(docs):
    return '+'.join(map(lambda x: x.link_title, filter(lambda d: len(d.link_title) > 1, docs)))


def map_docs_to_sentences(docs):
    data_map = {}
    stopwords = get_polish_stopwords()
    for key, doc_list in docs.items():
        txt = get_clean_text(get_sentences_with_mentions(get_word_tuples(doc_list)), stopwords)
        if len(txt) > 0:
            data_map[key] = txt
    return data_map


def save_to_file(filename, obj):
    with open(dir + filename + '.pickle', 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.close()


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


# def get_inv_mapping(_category_mappings, _type_jsons):
#     inv_map = {}
#     for key, value in _category_mappings.items():

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


def _find_by_entity(_entity, _mapping_tuples, _pl_map, _prefix_map, _lemma_map, _entities):
    if _entity in _pl_map:
        return _pl_map[_entity]

    # for _tuple in _mapping_tuples[_prefix_map[_entity[0]]:]:
    try:
        _index = _prefix_map[_entity[0]] if _entity[0] in _prefix_map else _prefix_map['z']
        for _tuple in _mapping_tuples[_index:_index + get_upper_bound(
                _mapping_tuples[_index], _mapping_tuples[_index:])]:
            try:
                if _tuple.similar_to(_entity, _entities):
                    return _tuple
            except AttributeError:
                i = 1
    except KeyError:
        j = 1
    return None


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


def strip_string(_entity):
    pattern = re.compile('[\W_]+')
    return pattern.sub('', _entity)


def strip_string_space(_entity):
    to_exclude = ['”', '„', '(', ')', '{', '}', '[', ']', '.', ',', '!', '?', '%', '\'', '\\', '/', '+', '-', '=',
                  ':', '*', '@', '#', '$', '^', '&', '_', '<', '>', '"', '÷', '’']
    _entity = _entity.lower().translate(
        {ord(i): None for i in to_exclude}) if _entity else False
    return _entity


def get_prefix_map(_entity_tuples):
    _prefix_map = {}
    cnt = 0
    for _e in _entity_tuples:
        if len(_e.cleaned_original_entity) > 0 and not _e.cleaned_original_entity[0] in _prefix_map:
            _prefix_map[_e.cleaned_original_entity[0]] = cnt
        cnt += 1
    return _prefix_map


def _get_label(_entity, _entity_tuples, _pl_map, _prefix_map, _lemma_map):
    # if ''.join(_entity.split()) in pl_map:
    #     return _pl_map[''.join(_entity.split())]
    _l = []
    for _e in _entity.split():
        _entity_lemma = get_entity(_e, _lemma_map)
        if _entity_lemma: _l.append(_entity_lemma)
    return find_tuple(_entity_tuples, _l, _lemma_map, _pl_map, _prefix_map)


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


def _strip_dangling_keywords(_entity):
    _l = _entity.split()
    _r = []
    for _i in range(0, int(len(_l) / 2)):
        for _j in range(int(len(_l) / 2), len(_l)):
            # if _i > _j:
            _e1 = strip_string(_l[_i])
            _e2 = strip_string(_l[_j])
            if levenshtein(_e1, _e2) > min(get_sensitivity(_e1), get_sensitivity(_e2)) and _l[_i] not in _r:
                _r.append(_l[_i])
            elif _l[_i] not in _r:
                _r.append(_l[_i])
    return ' '.join(_r)


def strip_dangling_keywords(_entity):
    _l = _entity.split()
    try:
        _r = [_l[0]]
    except IndexError:
        i = 1
    _checked = set()
    for _e in _l:
        then_append(_r, _e, _checked)
    return ' '.join(_r)


def then_append(_r, _e, _checked):
    for __e in _r:
        if levenshtein(__e, _e) > min(get_sensitivity(_e),
                                      get_sensitivity(__e)) and not contains(_r, _e):
            _r.append(_e)
            return


def contains(_r, _e):
    for __e in _r:
        if levenshtein(__e, _e) <= min(get_sensitivity(_e),
                                      get_sensitivity(__e)):
            return True
    return False


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


def get_entity(_e, _lemma_map, _stopwords):
    _t = strip_string(_e).lower()
    if _t and _t not in _stopwords:
        return _lemma_map[_t] if not _e.isdigit() and _t in _lemma_map else [_t]


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

# save_to_file("results-{}".format(multi), (found, not_found, errors))

# l = get_label('George Washington', entity_tuples, pl_map, prefix_map, lemma_map)
# l = get_label('Briana Kernighana ', entity_tuples, pl_map, prefix_map, lemma_map)
# l = get_label('asocjacyjna tablica ', entity_tuples, pl_map, prefix_map, lemma_map)
# l = get_label('tablica asocjacyjna', entity_tuples, pl_map, prefix_map, lemma_map)
# l = get_label('tablic asocjacyjnych ', entity_tuples, pl_map, prefix_map, lemma_map)
#
# w1 = find_by_entity('rolniczego', entity_tuples, pl_map, prefix_map, lemma_map)
# w2 = find_by_entity('rolniczy', entity_tuples, pl_map, prefix_map, lemma_map)

# t0 = find_by_entity('George Washington', entity_tuples, pl_map, prefix_map, lemma_map)
# t4 = find_by_entity('tablica asocjacyjna', entity_tuples, pl_map, prefix_map, lemma_map)
# t4 = find_by_entity('tablic asocjacyjnych', entity_tuples, pl_map, prefix_map, lemma_map)
#
# t1 = find_by_entity("a", entity_tuples, pl_map, prefix_map, lemma_map)
# t2 = find_by_entity("Brian Kernighan", entity_tuples, pl_map, prefix_map, lemma_map)

# category_mappings = get_pickled("mappings.true")

i = 1
