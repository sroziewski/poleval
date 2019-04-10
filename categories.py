import logging

from poleval.lib.poleval import get_pickled, map_docs_to_sentences, get_list_sentences

logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s", datefmt='%H:%M:%S', level=logging.INFO)



# input_file = dir + 'tokens-with-entities.tsv'
# input_file = dir + 'tokens-with-entities-and-tags_1mln.tsv'

# data_object_map(input_file, saved_data_file)



# page_object_map(pages_input_file, pages_output_file)
# article_parent_object_map(article_parents_input_file, article_parents_output_file)
# category_parent_object_map(category_parents_input_file, category_parents_output_file)
# child_article_object_map(child_articles_input_file, child_articles_output_file)
# child_category_object_map(child_categories_input_file, child_categories_output_file)
# link_by_source_object_map(link_by_source_input_file, link_by_source_output_file)

# saved_data_file = "1mln_tokens"


# save_to_file(saved_data_file, data)

def flatten_list(list):
    return [item for sublist in list for item in sublist]


def process_batches(sentences_out):
    for i in range(5, 7):
        data = get_pickled(saved_data_file.format(i))
        data_map_of_sentences = map_docs_to_sentences(data)
        sentences_for_docs = get_list_sentences(data_map_of_sentences)
        sentences = flatten_list(sentences_for_docs)
        sentences_out.append(sentences)


# data = get_pickled(saved_data_file)
# data = get_pickled(saved_data_file.format(5))
# data_filtered = filter_empty_docs(data)
# data_filtered_text = map_docs_to_text(data_filtered)
# data_map_of_sentences = map_docs_to_sentences(data)

# sentences_for_docs = get_list_sentences(data_map_of_sentences)
# sentences = flatten_list(sentences_for_docs)
# sentences = [item for sublist in sentences_for_docs for item in sublist]

# data_filtered_longer_tokens = filter_longer_tokens(data)

article_parents = get_pickled(article_parents_output_file)
pages = get_pickled(pages_output_file)
category_parents = get_pickled(category_parents_output_file)
child_articles = get_pickled(child_articles_output_file)
child_categories = get_pickled(child_categories_output_file)
# link_by_sources = get_pickled(link_by_source_output_file)
