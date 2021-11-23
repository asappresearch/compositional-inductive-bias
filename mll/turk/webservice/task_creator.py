"""
handles creating tasks
"""
import argparse
import os

from mll.turk.webservice import tasks


class TaskCreator:
    def __init__(self, task_type: str, grammar: str, seed: int, num_examples: int = 50):
        self.task_type = task_type
        self.seed = seed
        Task = getattr(tasks, task_type)
        self.task = Task(seed=seed, num_examples=num_examples, grammar=grammar)

    @property
    def max_cards(self):
        return self.task.max_cards

    def create_example(self, idx: int):
        return self.task.create_example(idx)


def run(args):
    if not os.path.isdir('html/img'):
        os.makedirs('html/img')
    task_creator = TaskCreator(**args.__dict__)
    ex = task_creator.create_example()
    os.system('open ' + ex['filepath'])
    print('ex', ex)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task-type', type=str, default='Comp1')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--idx', type=int, default=0)
    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()
