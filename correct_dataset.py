from poleval.lib.poleval import get_pickled

lemma_map = get_pickled("lemma_map_ext")
(category_map, entity_tuples, pl_map, en_map, disambiguation, prefix_map) = get_pickled("mapping-objects_ext")
(_found, _not_found, _errors) = get_pickled("results_chunks_try")

