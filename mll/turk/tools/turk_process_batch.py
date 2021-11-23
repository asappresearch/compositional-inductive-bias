import argparse
import csv
from os.path import expanduser as expand

import sqlalchemy
from sqlalchemy.orm import sessionmaker

from mll.turk.webservice import tables


def run(args):
    engine = sqlalchemy.create_engine(f'sqlite:///{expand(args.in_db)}', echo=False)
    print('engine', engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    print('session', session)
    print('')

    with open(expand(args.in_batch_csv), 'r') as f:
        dict_reader = csv.DictReader(f)
        batch = list(dict_reader)
    if args.out_processed_csv is not None:
        out_f = open(args.out_processed_csv, 'w')
        dict_writer = csv.DictWriter(out_f, fieldnames=dict_reader.fieldnames)
        dict_writer.writeheader()

    for ex in batch:
        try:
            completion_code = ex['Answer.surveycode'].replace('.', '')
            game_instance = session.query(tables.GameInstance).filter_by(
                completion_code=completion_code, task_id=args.task_id).first()
            if game_instance is not None:
                print(game_instance.feedback)
        except Exception as e:
            print('completion_code', completion_code)
            print(e)
            raise e
    print('')

    for ex in batch:
        completion_code = ex['Answer.surveycode'].replace('.', '')
        game_instance = session.query(tables.GameInstance).filter_by(
            completion_code=completion_code, task_id=args.task_id).first()
        if game_instance is None:
            print('rejecting', completion_code, ' => not found')
            ex['Reject'] = 'x'
            ex['RequesterFeedback'] = 'Cannot find your completion code in our database.'
        else:
            mins = game_instance.duration_seconds
            if mins is not None:
                mins //= 60
            print(
                args.task_id,
                'score', game_instance.score,
                'num_holdout_correct', game_instance.num_holdout_correct,
                'mins', mins,
                'num_cards', game_instance.num_cards)
            ex['Approve'] = 'x'
        if args.out_processed_csv is not None:
            dict_writer.writerow(ex)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in-db', default='pull/turk.db')
    parser.add_argument('--task-id', required=True)
    parser.add_argument('--in-batch-csv', type=str, required=True)
    # parser.add_argument('--out-processed-csv', type=str)
    # parser.add_argument('--process-all', action='store_true')
    args = parser.parse_args()
    args.out_processed_csv = args.in_batch_csv.replace('.csv', '') + '_proc.csv'
    run(args)
