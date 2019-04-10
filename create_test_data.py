import pickle


def get_pickled(filename):
    with open(dir + filename + '.pickle', 'rb') as handle:
        data = pickle.load(handle)
        handle.close()
        return data


def save_to_file(filename, obj):
    with open(dir + filename + '.pickle', 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.close()


def chunks(l, _n, _i):
    _step = int(len(l) / _n)
    if _i + 1 == _n:
        return l[_i * _step:]
    else:
        return l[_i*_step:(_i+1) * _step]


def create_chunks(_i):
    _test_tuples = get_pickled("test_tuples-{}".format(_i))
    for i in range(0, 5):
        _test_tuples_chunks = chunks(_test_tuples, 5, i)
        save_to_file("test_tuples_chunk_{}_{}".format(_i, i), _test_tuples_chunks)


for i in range(0, 19):
    create_chunks(1)