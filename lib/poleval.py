# coding=utf-8

import csv
import json
import sys

import editdistance
import pickle
import re
import requests
from bs4 import BeautifulSoup

from poleval.lib.definitions import dir
from poleval.lib.entity.structure import WordTriplet, Word, Entity, DataItem, LinkBySource, Page, ArticleParent, \
    CategoryParent, ChildArticle, ChildCategory, Mention


def levenshtein(s, t):
    return editdistance.eval(s, t)


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


def _get_text(docs):
    return ' '.join(map(lambda d: d.token, docs))


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
    if ''.join(_entity.split()) in _pl_map:
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


def data_object_map(input_file, output_file_prefix):
    docs_counter = 0
    multi_counter = 0
    input_data_map = {}
    docs_limit = 100000
    with open(input_file) as tsvfile:
        tsv_reader = csv.reader(tsvfile, delimiter="\t")
        for line in tsv_reader:
            if docs_counter == docs_limit:
                save_to_file(output_file_prefix.format(multi_counter), input_data_map)
                input_data_map = {}
                multi_counter += 1

            if len(line) > 0:
                data = DataItem(line)
                if data.doc_id in input_data_map:
                    input_data_map[data.doc_id].append(data)
                else:
                    input_data_map[data.doc_id] = [data]
            docs_counter = len(input_data_map)
        save_to_file(output_file_prefix.format(multi_counter), input_data_map)
    tsvfile.close()


def filter_empty_docs(docs):
    data_map = {}
    for key, doc_list in docs.items():
        data_map[key] = list(filter(lambda d: len(d.token) > 1, doc_list))
    return data_map


def link_by_source_object_map(input_file, output_file):
    input_data_map = {}
    with open(input_file) as csv_file:
        reader = csv.reader(csv_file, delimiter=",")
        for line in reader:
            if len(line) > 0:
                link_by_source = LinkBySource(line)
                input_data_map[link_by_source.source_id] = link_by_source
        csv_file.close()
        save_to_file(output_file, input_data_map)


def filter_longer_tokens(docs):
    data_map = {}
    for key, doc_list in docs.items():
        data_map[key] = list(filter(lambda d: len(d.token.split(' ')) > 1, doc_list))
    return data_map


def get_list_sentences(a_map):
    return [sentence for key, sentence in a_map.items()]


def page_object_map(input_file, output_file):
    input_data_map = {}
    with open(input_file) as csv_file:
        reader = csv.reader(csv_file, delimiter=",")
        for line in reader:
            if len(line) > 0:
                page = Page(line)
                input_data_map[page.page_id] = page
        csv_file.close()
        save_to_file(output_file, input_data_map)


def article_parent_object_map(input_file, output_file):
    input_data_map = {}
    with open(input_file) as csv_file:
        reader = csv.reader(csv_file, delimiter=",")
        for line in reader:
            if len(line) > 0:
                article_parent = ArticleParent(line)
                input_data_map[article_parent.article_id] = article_parent
        csv_file.close()
        save_to_file(output_file, input_data_map)


def category_parent_object_map(input_file, output_file):
    input_data_map = {}
    with open(input_file) as csv_file:
        reader = csv.reader(csv_file, delimiter=",")
        for line in reader:
            if len(line) > 0:
                category_parent = CategoryParent(line)
                input_data_map[category_parent.category_id] = category_parent
        csv_file.close()
        save_to_file(output_file, input_data_map)


def child_article_object_map(input_file, output_file):
    input_data_map = {}
    with open(input_file) as csv_file:
        reader = csv.reader(csv_file, delimiter=",")
        for line in reader:
            if len(line) > 0:
                child_article = ChildArticle(line)
                input_data_map[child_article.category_id] = child_article
        csv_file.close()
        save_to_file(output_file, input_data_map)


def child_category_object_map(input_file, output_file):
    input_data_map = {}
    with open(input_file) as csv_file:
        reader = csv.reader(csv_file, delimiter=",")
        for line in reader:
            if len(line) > 0:
                child_article = ChildCategory(line)
                input_data_map[child_article.category_id] = child_article
        csv_file.close()
        save_to_file(output_file, input_data_map)


def contains_digit(string):
    return any(i.isdigit() for i in string)


def clean_tuples(_list, _lemma_map, _stopwords):
    _new_list2 = []
    for _tuple in _list:
        _e1 = extract_tuple_text(_tuple[1], _lemma_map, _stopwords)
        _e1.extend(extract_tuple_text(_tuple[2], _lemma_map, _stopwords)) if _tuple[2] else None
        _new_list2.append((_tuple[0], set(flatten_list(_e1))))
    return _new_list2


def extract_tuple_text(_text, _lemma_map, _stopwords):
    _l1 = []
    for _t in _text.split():
        if _t:
            _ee = get_entity(_t, _lemma_map, _stopwords)
            if _ee:
                _l1.append(_ee)
    return _l1


def extract_existing_mapping(feats, _categories_dict):
    return list(filter(lambda x: x in _categories_dict, feats))


def get_ids(_id, type_id, _categories_dict, _global_map, _original_id, _breakable):
    features = get_features(get_divs(get_soup(_id), type_id))
    found_mapping = extract_existing_mapping(features, _categories_dict)
    if len(found_mapping) > 0:
        _global_map[_original_id] = found_mapping[0]
        _breakable.append(True)
        return

    for feat in features:
        if feat not in _categories_dict and len(_breakable) == 0:
            get_ids(feat, "P31", _categories_dict, _global_map, _original_id, _breakable)
            get_ids(feat, "P279", _categories_dict, _global_map, _original_id, _breakable)


def get_features(divs):
    if not divs:
        return []
    divs_instances = divs.findAll("div", {"class": "wikibase-snakview-variation-valuesnak"})
    instances = []
    for div_instance in divs_instances:
        if div_instance.findAll("a") and 'title' in div_instance.findAll("a")[0].attrs:
            instances.append(div_instance.findAll("a")[0].attrs['title'])
    return instances


def get_subclasses(divs_subclasses):
    subclasses = []
    for div_instance in divs_subclasses:
        subclasses.append(div_instance.findAll("a")[0].attrs['title'])
    return subclasses


def get_context2(tuples):
    context = []
    sentences = []
    handy = []
    flag = 0
    is_entity = False
    for _ in tuples:
        if _.type == 'interp' and _.lemma == '.':
            if flag == 1:
                context.append(handy)
                flag = 0
            if is_entity:
                if len(sentences) - 1 >= 0 and sentences[len(sentences) - 1] not in context:
                    context.append(sentences[len(sentences) - 1])
                if handy not in context:
                    context.append(handy)
                flag = 1
            sentences.append(handy)
            handy = []
            is_entity = False
        elif _.type != 'interp':
            if _.is_entity:
                is_entity = True
            handy.append(_.lemma.split(':')[0].lower().translate(
                {ord(i): None for i in ['”', '„', '(', ')', '{', '}', '[', ']']}))
    if handy:
        sentences.append(handy)
        if is_entity:
            context.append(sentences[len(sentences - 1)])
            context.append(handy)
    return context


def _map_docs_to_sentences(docs):
    data_map = {}
    stopwords = get_polish_stopwords()
    for key, doc_list in docs.items():
        txt = get_clean_text(get_sentences_with_mentions(get_word_tuples(doc_list)), stopwords)
        if len(txt) > 0:
            data_map[key] = txt
    return data_map


def categories_to_vectors(_model, _categories):
    _category_vectors = []
    for category in _categories:
        if len(category.split()) > 1:
            vec = (_model.wv.get_vector(category.split()[0]) + _model.wv.get_vector(
                category.split()[1])) / 2
        else:
            vec = _model.wv.get_vector(category)
        _category_vectors.append(vec)
    return _category_vectors


class recursion_limit:
    def __init__(self, limit):
        self.limit = limit
        self.old_limit = sys.getrecursionlimit()

    def __enter__(self):
        sys.setrecursionlimit(self.limit)

    def __exit__(self, type, value, tb):
        sys.setrecursionlimit(self.old_limit)


def filter_out_blinds(features):
    return list(filter(lambda x: x not in (
        'Q55983715', 'Q23958852', 'Q24017414', 'Q24017465', 'Q16889133', 'Q21146257', 'Q17538690', 'Q83306', 'Q328'),
                       features))


def get_outsiders(jsons, types):
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
    return list(map(lambda x: x['id'], _outsiders))


def get_lemma_map(data):
    _lemma_map = {}
    for _data_item_list in data.values():
        for _data_item in _data_item_list:
            to_exclude = [' ', '”', '„', '(', ')', '{', '}', '[', ']', '.', ',', '!', '?', '%', '\'', '\\', '/', '+',
                          '-', '=',
                          ':', '*', '@', '#', '$', '^', '&', '_', '<', '>', '"', '÷']
            _entity = _data_item.token.lower().translate(
                {ord(i): None for i in to_exclude}) if _data_item.token else False
            _lemma_map[_entity] = _data_item.lemma
    return _lemma_map
