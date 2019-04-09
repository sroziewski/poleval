from poleval.lib.poleval import data_object_map, get_pickled, get_test_data

input_file = dir + 'test/task3_test.tsv'
saved_data_file = "test/task3_test"

data_object_map(input_file, saved_data_file)
data = get_pickled(saved_data_file)
test_tuples = get_test_data(data)

(category_map, entity_tuples, pl_map, en_map, disambiguation, prefix_map) = get_pickled("mapping-objects_ext")
lemma_map = get_pickled("lemma_map_ext")


