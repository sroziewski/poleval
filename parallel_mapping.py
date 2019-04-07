import json
import requests
import pickle
from bs4 import BeautifulSoup
import sys
import multiprocessing
from  multiprocessing import Pool, Queue
from time import time
from multiprocessing import Process, Manager
from multiprocessing.managers import BaseManager

dir = '/home/szymon/juno/challenge/poleval/'


class recursion_limit:
    def __init__(self, limit):
        self.limit = limit
        self.old_limit = sys.getrecursionlimit()

    def __enter__(self):
        sys.setrecursionlimit(self.limit)

    def __exit__(self, type, value, tb):
        sys.setrecursionlimit(self.old_limit)


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


def get_soup(id):
    r = requests.get("https://www.wikidata.org/wiki/" + id)
    return BeautifulSoup(r._content, 'html.parser')


def get_divs(_soup, type_id):
    r = _soup.findAll("div", {"id": type_id})
    if r:
        return r[0]
    else:
        return None


def filter_out_blinds(features):
    return list(filter(lambda x: x not in (
        'Q55983715', 'Q23958852', 'Q24017414', 'Q24017465', 'Q16889133', 'Q21146257', 'Q17538690', 'Q83306', 'Q328'),
                       features))


def to_terminate(_original_id, _type_id, lp31, lp279, lp17, _breakable):
    _level = 20
    if _type_id == 'P31' and lp31 > _level:
        _breakable.append(True)
        print("{} {}".format(_type_id, _original_id))
        return True
    if _type_id == 'P279' and lp279 > _level:
        _breakable.append(True)
        print("{} {}".format(_type_id, _original_id))
        return True
    if _type_id == 'P17' and lp17 > _level:
        _breakable.append(True)
        print("{} {}".format(_type_id, _original_id))
        return True
    return False


def get_ids(_id, _type_id, _categories_dict, _global_map, _original_id, _breakable, _recursion_level_p31,
            _recursion_level_p279, _recursion_level_p17):
    if to_terminate(_original_id, _type_id, _recursion_level_p31, _recursion_level_p279, _recursion_level_p17,
                    _breakable):
        return
    if len(_breakable) > 0:
        return
    features = filter_out_blinds(get_features(get_divs(get_soup(_id), _type_id)))
    found_mapping = extract_existing_mapping(features, _categories_dict)
    if len(found_mapping) > 0:
        _global_map[_original_id] = found_mapping[0]
        _breakable.append(True)
        return

    for feat in features:
        if feat not in _categories_dict and len(_breakable) == 0:
            get_ids(feat, "P279", _categories_dict, _global_map, _original_id, _breakable, _recursion_level_p31,
                    _recursion_level_p279 + 1, _recursion_level_p17)
            get_ids(feat, "P31", _categories_dict, _global_map, _original_id, _breakable, _recursion_level_p31 + 1,
                    _recursion_level_p279, _recursion_level_p17)
            get_ids(feat, "P17", categories_dict, _global_map, _original_id, _breakable, _recursion_level_p31,
                    _recursion_level_p279, _recursion_level_p17 + 1)


def extract_existing_mapping(feats, _categories_dict):
    return list(filter(lambda x: x in _categories_dict, feats))


def get_mapping(_id, _global_map):
    original_id = _id
    breakable = []
    get_ids(original_id, "P279", categories_dict, _global_map, original_id, breakable, 0, 0, 0)
    if len(breakable) == 0:
        breakable = []
        try:
            get_ids(original_id, "P31", categories_dict, _global_map, original_id, breakable, 0, 0, 0)
            if len(breakable) == 0:
                breakable = []
                get_ids(original_id, "P17", categories_dict, _global_map, original_id, breakable, 0, 0, 0)
        except RuntimeError:
            print(original_id)


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
    def __init__(self, lemma, type, entity):
        self.lemma = lemma
        self.type = type
        self.is_entity = entity


class Mention(object):
    def __init__(self, sentence, with_entity):
        self.sentence = sentence
        self.with_entity = with_entity


def get_pickled(filename):
    with open(dir + filename + '.pickle', 'rb') as handle:
        data = pickle.load(handle)
        handle.close()
        return data


def cleaning(doc_list, stopwords):
    # Lemmatizes and removes stopwords
    # doc needs to be a spacy Doc object
    txt = [token.strip() for token in doc_list if not token.strip() in stopwords]
    # Word2Vec uses context words to learn the vector representation of a target word,
    # if a sentence is only one or two words long,
    # the benefit for the training is very small
    if len(txt) > 2:
        return txt


def get_clean_text(token_lists, stopwords):
    return [x for x in [cleaning(tokens, stopwords) for tokens in token_lists] if x is not None]


def get_polish_stopwords():
    with open("/juno/challenge/poleval/polish.stopwords.txt") as file:
        return [line.strip() for line in file]


def get_word_tuples(docs):
    tuples = []
    for doc in docs:
        value = doc.lemma
        if len(doc.link_title) > len(doc.lemma):
            value = doc.link_title
        tuples.append(WordTriplet(value, doc.morphosyntactic_tags, doc.entity_id != '_'))
    return tuples


def get_sentences_with_mentions(tuples):
    sentences = []
    handy = []
    is_entity = False

    for _ in tuples:
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


def save_to_file(filename, obj):
    with open(dir + filename + '.pickle', 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.close()


def map_docs_to_sentences(docs):
    data_map = {}
    stopwords = get_polish_stopwords()
    for key, doc_list in docs.items():
        txt = get_clean_text(get_sentences_with_mentions(get_word_tuples(doc_list)), stopwords)
        if len(txt) > 0:
            data_map[key] = txt
    return data_map


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


def extract_main_entity_category2(jsons, types, _category_vectors, _model_v2w, _categories, categories_dict):
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
    return _outsiders


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


input_file = dir + 'tokens-with-entities-and-tags_1mln.tsv'
saved_data_file = "20000/tokens-with-entities-and-tags_1mln"
json_file = dir + 'entities.jsonl'
entity_types_file = dir + 'entity-types.tsv'
entity_types_file_output = 'entity-types'

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

categories = ["człowiek", "położenie geograficzny", "dyscyplina naukowy", "struktura anatomiczny", "zajęcie",
              "model pojazd", "konstrukcja", "utwór pisany", "ciało niebieskie", "odzież", "takson", "byt mityczny",
              "dyscyplina sport", "istota nadprzyrodzony", "ciecz", "system polityczny", "zwierzę",
              "substancja chemiczny", "publikacja", "ukształtowanie teren", "język", "jednostka", "obiekt geograficzny",
              "dzieło artystyczny",
              "narzędzie", "organizm", "jedzenie"]

# entity_types = get_entity_types(entity_types_file)
# / type_jsons = read_json_file(json_file)
# / save_to_file(entity_types_file_output, type_jsons)
# type_jsons = get_pickled(entity_types_file_output)
#
#
# a = w2vec_model_3.wv.cosine_similarities((w2vec_model_3.wv.get_vector("kraj")+w2vec_model_3.wv.get_vector("federacja")+w2vec_model_3.wv.get_vector("niepodległy")+w2vec_model_3.wv.get_vector("kolonialny"))/4,  category_vectors)
#
# outsiders = get_outsiders(type_jsons, entity_types)
# save_to_file("outsiders", outsiders)

outsiders = get_pickled("outsiders")


# global_map = {}


class GlobalMapClass(object):
    def __init__(self):
        self.map = {}

    def set(self, value):
        self.map = value

    def get(self):
        return self.map


# get_mapping("Q23165129", global_map)

from functools import partial
from multiprocessing.pool import ThreadPool


def mp_worker(_id, _shared_map):
    t = time()
    get_mapping(_id, _shared_map.get())
    print('Time to train the model: {} mins'.format(round((time() - t) / 60, 2)))
    return _shared_map


def mp_handler(ids, _shared_map):
    pool = ThreadPool(1)
    # lock = multiprocessing.Lock()
    pool.apply(mp_worker, args=(ids, _shared_map))
    pool.close()
    pool.join()


BaseManager.register('GlobalMapClass', GlobalMapClass)
manager = BaseManager()
manager.start()
shared_map = manager.GlobalMapClass()

mp_handler(outsiders[0], shared_map)

i = 1
# with recursion_limit(1500):
#     global_map = {}
#     for _id in outsiders[:100000]:
#         get_mapping(_id, global_map)
#
# save_to_file("global_map.1", global_map)
