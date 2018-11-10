from Charm.utils.preprocessing import IOHelper

import itertools


class App(object):
    def __init__(self, tag, f, c=0):
        self.set_f(f)
        self.set_c(c)
        self.tag = tag

    def set_f(self, f):
        self.f = f

    def set_c(self, c):
        self.c = c

    def get_printable(self):
        return '{} ({}, {})'.format(self.tag, self.f, self.c)

    def gen_feed(self):
        feed_f = ('f', self.f)
        feed_c = ('c', self.c)
        return [feed_f, feed_c]

class AppHelper(object):
    @staticmethod
    def gen_app(path=None, regress_model=None):
        fs = [.5, .9, .99, .999]
        cs = [.0, .001, .01, .1]
        tag = 'Pathological'
        if path is not None:
            assert(regress_model)
            data, wklds = IOHelper.read_data(path)
        else:
            return [App(tag, f, c) for f, c in itertools.product(fs, cs)]

        if regress_model:
            apps = []
            for k, df in data.groupby('workload'):
                cores = df['thread_count']
                speedup = df['speedup']
                result = regress_model.fit(cores, speedup)
                if 'c' in result: 
                    apps.append(App(k, result['f'], result['c']))
                else:
                    apps.append(App(k, result['f'], 0))
            return apps
