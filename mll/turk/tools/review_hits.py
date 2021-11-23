import argparse
from os.path import expanduser as expand
from ruamel import yaml

import sqlalchemy
from sqlalchemy.orm import sessionmaker

from mll.turk.webservice import tables
import turk_lib


def yaml_dumps(d):
    with open('/tmp/foo.yaml', 'w') as f:
        yaml.safe_dump(d, f)
    with open('/tmp/foo.yaml', 'r') as f:
        contents = f.read()
    return contents


def run(args):
    engine = sqlalchemy.create_engine(f'sqlite:///{expand(args.in_db)}', echo=False)
    print('engine', engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    print('session', session)
    print('')

    turk = turk_lib.get_turk(prod=args.prod)
    # response = turk.list_reviewable_hits(MaxResults=100)
    # response = turk.list_reviewable_hits(MaxResults=100, NextToken=response['NextToken'])
    # print(json.dumps(response, indent=2))
    hits = turk.list_hits(MaxResults=100)['HITs']
    # for hit in response['HITs']:
    for hit in hits:
        hit_id = hit['HITId']
        if 'RequesterAnnotation' not in hit:
            print('missing RequesterAnnotation')
            continue
        batch_id = hit['RequesterAnnotation'].split(':')[1].split(';')[0]
        template_id = hit['RequesterAnnotation'].split('OriginalHitTemplateId:')[1].split(';')[0]
        print('hit_id', hit_id, 'batch_id', batch_id, 'template_id', template_id)
        hit = turk.get_hit(HITId=hit_id)
        # print('hit', json.dumps(hit, indent=2))
        # print('hit', yaml_dumps(hit))
        question = hit['HIT']['Question']
        # print('question', question)
        task_id = question.split('.com/web/index.html?taskId=')[1].split('"')[0]
        print('task_id', task_id)
        print(f'https://requester.mturk.com/batches/{batch_id}/results')
        print('')
        assignment_response = turk.list_assignments_for_hit(HITId=hit_id, AssignmentStatuses=['Submitted'])
        assignments = assignment_response['Assignments']
        for assignment in assignments:
            print('')
            answer = assignment['Answer']
            worker_id = assignment['WorkerId']
            # status = assignment['AssignmentStatus']
            """<?xml version="1.0" encoding="ASCII"?>
            <QuestionFormAnswers
            xmlns="http://mechanicalturk.amazonaws.com/AWSMechanicalTurkDataSchemas/2005-10-01/QuestionFormAnswers.xsd">
            <Answer>
            <QuestionIdentifier>surveycode</QuestionIdentifier>
            <FreeText>24bf55fc3c184ebc8023c23ed19478de</FreeText>
            </Answer>
            </QuestionFormAnswers>
            """
            completion_code = answer.split('FreeText>')[1].split('<')[0].strip().replace('.', '')
            game_instance = session.query(tables.GameInstance).filter_by(
                completion_code=completion_code).first()
            if game_instance is not None:
                if game_instance.status != 'COMPLETE':
                    continue
                print(
                    'task_id', task_id, 'completion_code', completion_code)
                print('requester_id', game_instance.requester_id)
                print(
                    'score', game_instance.score,
                    'num_holdout_correct', game_instance.num_holdout_correct,
                    'duration', game_instance.duration_seconds,
                    'num_cards', game_instance.num_cards
                )
                print(game_instance.feedback)
                if task_id == game_instance.task_id:
                    print('task match', task_id)
                else:
                    print('TASK MISMATCH intended=' + task_id, 'actual=' + game_instance.task_id)
                score = game_instance.score
                if score is None:
                    score = 0
                cents_per_ten = game_instance.cents_per_ten
                if cents_per_ten is None:
                    cents_per_ten = 0
                else:
                    cents_per_ten = cents_per_ten
                bonus = (cents_per_ten * score / 10) / 100
                bonus_str = '%.2f' % (bonus)
                print('bonus_str', bonus_str, 'worker_id', worker_id)
            else:
                score = None
                print('cannot find for completion code', completion_code)
                print(assignment['AssignmentStatus'], assignment['SubmitTime'])
                if len(completion_code) == len('932cbb2d86eb45a2804831b283efadce'):
                    continue

            if score is None or score < 100:
                step_results = session.query(tables.StepResult).filter_by(
                    requester_id=game_instance.requester_id, task_id=task_id
                )
                for step_result in step_results:
                    print(step_result.expected_utt, step_result.player_utt, step_result.score)
                decision = input('decision (a/n/s) > ')
                if decision == 'n':
                    print('REJECT')
                    reason = input('Reason > ')
                    print('reason', reason)
                    assert reason.strip() != ''
                elif decision == 'a':
                    print('ACCEPT')
                else:
                    print('skipping')
                    continue
            else:
                decision = 'a'
            assignment_id = assignment['AssignmentId']
            if args.commit:
                if decision == 'n':
                    print(turk.reject_assignment(AssignmentId=assignment_id, RequesterFeedback=reason))
                else:
                    print(turk.approve_assignment(AssignmentId=assignment_id))
                    if bonus > 0:
                        print(turk.send_bonus(
                            AssignmentId=assignment_id, BonusAmount=bonus_str, WorkerId=worker_id,
                            Reason=(
                                f'{game_instance.score} points converted to bonus at '
                                f'{cents_per_ten} cents per 10 points.')))
                        bonuses = turk.list_bonus_payments(
                            # HITId=hit_id,
                            AssignmentId=assignment_id,
                            MaxResults=100
                        )
                        for bonus in bonuses['BonusPayments']:
                            if bonus['WorkerId'] == worker_id:
                                print(bonus['BonusAmount'], bonus['Reason'])
            print('')
        # if len(assignments) > 0:
        #     break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prod', action='store_true')
    parser.add_argument('--in-db', default='pull/turk.db')
    parser.add_argument('--commit', action='store_true')
    args = parser.parse_args()
    run(args)
