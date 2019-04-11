from gensim.models import Word2Vec

from poleval.lib.entity.definitions import categories, categories_dict, entity_types_file, entity_types_file_output, \
    saved_data_file_tokens_entities_tags
from poleval.lib.poleval import map_docs_to_sentences, get_pickled, extract_main_entity_category


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



w2vec_model_3 = Word2Vec.load(dir + "all-sentences-word2vec-m3.model")


category_vectors = categories_to_vectors(w2vec_model_3, categories)


entity_types = get_entity_types(entity_types_file)
# type_jsons = read_json_file(json_file)
# save_to_file(entity_types_file_output, type_jsons)
type_jsons = get_pickled(entity_types_file_output)


a = w2vec_model_3.wv.cosine_similarities((w2vec_model_3.wv.get_vector("kraj")+w2vec_model_3.wv.get_vector("federacja")+w2vec_model_3.wv.get_vector("niepodległy")+w2vec_model_3.wv.get_vector("kolonialny"))/4,  category_vectors)

extract_main_entity_category(type_jsons, entity_types, category_vectors, w2vec_model_3, categories, categories_dict)



data = get_pickled(saved_data_file_tokens_entities_tags)
data_map_of_sentences = map_docs_to_sentences(data)

