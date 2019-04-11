from gensim.models.phrases import Phrases

import logging

from poleval.lib.entity.definitions import saved_data_file_tokens_entities_tags
from poleval.lib.poleval import WordTuple, get_pickled, get_lemma_map, save_to_file, map_docs_to_sentences, \
    flatten_list, get_mentions

logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s", datefmt='%H:%M:%S', level=logging.INFO)


def get_text(docs):
    return ' '.join(map(lambda d: d.token, docs))


def get_word_tuples(docs):
    tuples = []
    for doc in docs:
        value = doc.lemma
        if len(doc.link_title) > len(doc.lemma):
            value = doc.link_title
        tuples.append(WordTuple(value, doc.morphosyntactic_tags))
    return tuples



def map_docs_to_mentions(docs):
    data_map = {}
    for key, doc_list in docs.items():
        data_map[key] = get_mentions(doc_list)
    return data_map


def get_list_sentences(a_map):
    return [sentence for key, sentence in a_map.items()]


def get_bigram_transformer(sentences):
    # for sentence in sentences:
    #     Phrases([row.split() for row in sentences], min_count=30, progress_per=10000)
    return Phrases([sentence for sentence in sentences], min_count=30, progress_per=10000)


def process_batches_for_lemma():
    for i in range(0, 19):
        data = get_pickled(saved_data_file_tokens_entities_tags.format(i))
        lemma_map = get_lemma_map(data)
    save_to_file("lemma_map", lemma_map)


def process_batches(sentences_out):
    for i in range(0, 19):
        data = get_pickled(saved_data_file_tokens_entities_tags.format(i))
        data_map_of_sentences = map_docs_to_sentences(data)
        sentences_for_docs = get_list_sentences(data_map_of_sentences)
        sentences = flatten_list(sentences_for_docs)
        sentences_out.append(sentences)

# page_object_map(pages_input_file, pages_output_file)
# article_parent_object_map(article_parents_input_file, article_parents_output_file)
# category_parent_object_map(category_parents_input_file, category_parents_output_file)
# child_article_object_map(child_articles_input_file, child_articles_output_file)
# child_category_object_map(child_categories_input_file, child_categories_output_file)
# link_by_source_object_map(link_by_source_input_file, link_by_source_output_file)

# saved_data_file = "1mln_tokens"


# save_to_file(saved_data_file, data)


process_batches_for_lemma()

sentences_pred = []
process_batches(sentences_pred)
sentences = flatten_list(sentences_pred)

save_to_file("all-sentences", sentences)
