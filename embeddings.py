import pickle
import json
from gensim.models import Word2Vec

from poleval.lib.poleval import get_polish_stopwords, get_clean_text, get_word_tuples, map_docs_to_sentences, \
    get_pickled


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
    i = 1


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


categories_dict = {'Q5':"człowiek", 'Q2221906':"położenie geograficzne", 'Q11862829':"dyscyplina naukowa", 'Q4936952':"struktura anatomiczna", 'Q12737077':"zajęcie", 'Q29048322':"model pojazdu", 'Q811430':"konstrukcja",
              'Q47461344':"utwór pisany", 'Q6999':"ciało niebieskie", 'Q11460':"odzież", 'Q16521':"takson", 'Q24334685':"byt mityczny", 'Q31629':"dyscyplina sportu",
              'Q28855038':"istota nadprzyrodzona", 'Q11435':"ciecz", 'Q28108':"system polityczny", 'Q16334298':"zwierzę", 'Q43460564':"substancja chemiczna", 'Q732577':"publikacja",
              'Q271669':"ukształtowanie terenu", 'Q34770':"język", 'Q2198779':"jednostka", 'Q20719696':"obiekt geograficzny", 'Q15621286':"dzieło artystyczne", 'Q39546':"narzędzie", 'Q7239':"organizm", 'Q2095':"jedzenie"}

categories = ["człowiek", "położenie geograficzny", "dyscyplina naukowy", "struktura anatomiczny", "zajęcie", "model pojazd", "konstrukcja", "utwór pisany", "ciało niebieskie", "odzież", "takson", "byt mityczny",
              "dyscyplina sport", "istota nadprzyrodzony", "ciecz", "system polityczny", "zwierzę", "substancja chemiczny", "publikacja", "ukształtowanie teren", "język", "jednostka", "obiekt geograficzny", "dzieło artystyczny",
              "narzędzie", "organizm", "jedzenie"]

w2vec_model_3 = Word2Vec.load(dir + "all-sentences-word2vec-m3.model")


category_vectors = categories_to_vectors(w2vec_model_3, categories)


entity_types = get_entity_types(entity_types_file)
# type_jsons = read_json_file(json_file)
# save_to_file(entity_types_file_output, type_jsons)
type_jsons = get_pickled(entity_types_file_output)


a = w2vec_model_3.wv.cosine_similarities((w2vec_model_3.wv.get_vector("kraj")+w2vec_model_3.wv.get_vector("federacja")+w2vec_model_3.wv.get_vector("niepodległy")+w2vec_model_3.wv.get_vector("kolonialny"))/4,  category_vectors)

extract_main_entity_category(type_jsons, entity_types, category_vectors, w2vec_model_3, categories, categories_dict)



data = get_pickled(saved_data_file)
data_map_of_sentences = map_docs_to_sentences(data)

