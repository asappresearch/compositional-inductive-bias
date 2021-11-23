class Stats(object):
    def __init__(self, keys):
        self._keys = keys
        self.reset()

    def __repr__(self):
        res = ''
        for k, v in self.__dict__.items():
            if k.startswith('_'):
                continue
            res += '  ' + str(k) + ':' + str(v) + '\n'
        return res

    def reset(self):
        for k in self._keys:
            self.__dict__[k] = 0.0
        self.its = 0

    def __iadd__(self, second):
        assert isinstance(second, Stats)
        for k, v in second.__dict__.items():
            if k.startswith('_'):
                continue
            self.__dict__[k] += second.__dict__[k]
        return self

    def incr(self, key: str, amount):
        assert key in self._keys
        self.__dict__[key] += amount

    def add_it(self):
        self.its += 1

    def to_means(self):
        means = {}
        for k in self._keys:
            k_mean = k.replace('_sum', '')
            means[k_mean] = self.__dict__[k] / self.its
        return means
