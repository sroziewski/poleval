from poleval.lib.definitions import saved_data_file_test_set, input_file_test_set
from poleval.lib.poleval import data_object_map, get_pickled, get_test_data

data_object_map(input_file_test_set, saved_data_file_test_set)
data = get_pickled(saved_data_file_test_set)
test_tuples = get_test_data(data)

(category_map, entity_tuples, pl_map, en_map, disambiguation, prefix_map) = get_pickled("mapping-objects_ext")
lemma_map = get_pickled("lemma_map_ext")


