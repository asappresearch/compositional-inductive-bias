import argparse
from datetime import datetime

import turk_lib


def run(args):
    turk = turk_lib.get_turk(prod=args.prod)

    for item in turk.list_hits()['HITs']:
        hit_id = item['HITId']
        print('HITId:', hit_id)

        status = turk.get_hit(HITId=hit_id)['HIT']['HITStatus']
        print('HITStatus:', status)

        if status == 'Assignable':
            turk.update_expiration_for_hit(
                HITId=hit_id,
                ExpireAt=datetime(2015, 1, 1)
            )

        try:
            turk.delete_hit(HITId=hit_id)
        except Exception as e:
            print(e)
            print('Not deleted')
        else:
            print('Deleted')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prod', action='store_true')
    args = parser.parse_args()
    run(args)
