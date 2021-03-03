import time
import torch
from collections import defaultdict


class Timer(object):
    """
    call 'state' *after* each thing you're interested in measuring, with the name of that thing

    eg:

    timer.reset()
    do_a()
    timer.state('a')
    do_b()
    timer.state('b')
    timer.dump()
    """
    def __init__(self):
        self.reset()

    def reset(self):
        torch.cuda.synchronize()
        self.times_by_state = defaultdict(float)
        # self.current_state = '__init__'
        self.last_time = time.time()

    def state(self, state):
        torch.cuda.synchronize()
        self.times_by_state[state] += (time.time() - self.last_time)
        # self.current_state = state
        self.last_time = time.time()

    def dump(self):
        times = [{'name': name, 'time': time} for name, time in self.times_by_state.items()]
        times.sort(key=lambda x: x['time'], reverse=True)
        print('timings:')
        for t in times:
            print('    ', t['name'], '%.3f' % t['time'])
        self.reset()
