import os
import time
from os import path
import subprocess

import torch


def ensure_dirs_exist(dirs):
    for d in dirs:
        if not path.isdir(d):
            os.makedirs(d)
            print('created directory [%s]' % d)


def safe_save(filepath, target):
    """
    if out of disk space, will crash instead of erasing existing file
    """
    print('saving... ', end='', flush=True)
    save_start = time.time()
    with open(filepath + '.tmp', 'wb') as f:
        torch.save(target, f)
    os.rename(filepath + '.tmp', filepath)
    print('saved in %.1f seconds' % (time.time() - save_start))


def get_date_ordered_files(target_dir):
    files = subprocess.check_output(['ls', '-rt', target_dir]).decode('utf-8').split('\n')
    files = [f for f in files if f != '']
    return files
