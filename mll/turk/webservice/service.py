"""
for ssl, use nginx, and get cert as per
https://www.nginx.com/blog/using-free-ssltls-certificates-from-lets-encrypt-with-nginx/

example config

    server {
        server_name           <url to server here>;
        client_max_body_size  200M;

        ## Main site location.
        location / {
            proxy_pass                          http://127.0.0.1:5000;
            proxy_set_header                    Host $host;
            proxy_set_header X-Forwarded-Host   $server_name;
            proxy_set_header X-Real-IP          $remote_addr;
        }

    listen 443 ssl; # managed by Certbot
    ssl_certificate /etc/letsencrypt/live/<url to server here>/fullchain.pem; # managed by Certbot
    ssl_certificate_key /etc/letsencrypt/live/<url to server here>/privkey.pem; # managed by Certbot
    include /etc/letsencrypt/options-ssl-nginx.conf; # managed by Certbot
    ssl_dhparam /etc/letsencrypt/ssl-dhparams.pem; # managed by Certbot

}

    server {
    if ($host = <url to server here>) {
        return 301 https://$host$request_uri;
    } # managed by Certbot
        listen  80 default_server;
        server_name           <url to server here>;
    return 404; # managed by Certbot
}
"""
import datetime
import argparse
import uuid
import random
from typing import List, Optional
import os
from aiohttp import web

from aiohttp_cors.resource_options import ResourceOptions
from ruamel import yaml
import sqlalchemy
from sqlalchemy.orm import sessionmaker
import aiohttp_cors
import numpy as np

from mll.turk.webservice.task_creator import TaskCreator
from mll.turk.webservice import tables, datetime_utils


num_examples_per_game = 50


def get_unique_string():
    return uuid.uuid4().hex


def meaning_to_str(meaning: List[int]):
    return ','.join([str(v) for v in meaning])


def create_game_instance(
        remote, requester_id: str, task_id: str, task_type: str, grammar: str, num_holdout: int, config):
    seed = r.randint(1000000)
    task_creator = TaskCreator(
        task_type=task_type, seed=seed, grammar=grammar,
        num_examples=num_examples_per_game)
    game_instance = tables.GameInstance(
        requester_id=requester_id, task_id=task_id, example_idx=0, seed=seed, num_steps=num_examples_per_game,
        start_datetime=datetime_utils.datetime_to_str(datetime.datetime.now()), status='STARTED',
        completion_code=get_unique_string(), max_cards=task_creator.max_cards, remote=remote, num_cards=2,
        num_holdout=num_holdout, num_holdout_correct=0, cents_per_ten=config.cents_per_ten)
    return game_instance


def get_config(task_id: str):
    with open(f'mll/turk/webservice/configs/{task_id}.yaml') as f:
        config = yaml.safe_load(f)
    # print('config', config)
    config['num_holdout'] = config.get('num_holdout', 0)
    config['holdout_idxes'] = config.get('holdout_idxes', [])
    config['cents_per_ten'] = config.get('cents_per_ten', 0)
    # print('config', config)
    return argparse.Namespace(**config)


class AutoInc:
    def __init__(self, autoinc_every: Optional[int], game_instance):
        self.autoinc_every = autoinc_every
        self.game_instance = game_instance

    def __call__(self, example_idx):
        if self.autoinc_every is not None:
            if (example_idx + 1) % self.autoinc_every == 0:
                _min_num_cards = (example_idx + 1) // self.autoinc_every + 2
                _min_num_cards = min(self.game_instance.max_cards - self.game_instance.num_holdout, _min_num_cards)
                if _min_num_cards > self.game_instance.num_cards:
                    self.game_instance.num_cards += 1


def handle_completion(game_instance):
    end_datetime = datetime.datetime.now()
    game_instance.finish_datetime = datetime_utils.datetime_to_str(end_datetime)
    game_instance.status = 'COMPLETE'
    start_datetime = datetime_utils.str_to_datetime(game_instance.start_datetime)
    duration_seconds = datetime_utils.datetime_diff_seconds(end_datetime, start_datetime)
    game_instance.duration_seconds = duration_seconds
    session.commit()
    return web.json_response({
        'messageType': 'gameCompleted',
        'taskId': game_instance.task_id,
        'requesterId': game_instance.requester_id,
        'score': game_instance.score,
        'completionCode': game_instance.completion_code
    })


def create_new_step_result(config, game_instance):
    task_creator = TaskCreator(
        task_type=config.task_type, seed=game_instance.seed,
        num_examples=num_examples_per_game, grammar=config.grammar)
    is_holdout = False
    # print('holdout_idxes', config.holdout_idxes)
    if game_instance.example_idx in config.holdout_idxes:
        # print('creating holdout example')
        # holdout example
        holdout_idx = config.holdout_idxes.index(game_instance.example_idx)
        idx = game_instance.num_cards + holdout_idx
        is_holdout = True
    else:
        idx = random.randint(0, game_instance.num_cards - 1)
    # print('num_cards', game_instance.num_cards, 'idx', idx)
    ex = task_creator.create_example(idx=idx)
    # print('ex', ex)

    start_datetime = datetime_utils.datetime_to_str(datetime.datetime.now())
    step_result = tables.StepResult(
        requester_id=game_instance.requester_id, task_id=game_instance.task_id,
        example_idx=game_instance.example_idx, image_path=ex['filepath'],
        expected_utt=ex['expected'], meaning=meaning_to_str(ex['meaning']), start_datetime=start_datetime,
        status='TODO', score=0, num_cards=game_instance.num_cards, is_holdout=is_holdout)
    return step_result


async def _fetch(request_object, increment_idx: bool = False):
    """
    Expected request, example:

    {
        requesterId: "ABCEEFGCDF",
        taskId: "COMP",
    }

    response example:
    {
        pictureUrl: "img/asdevsafdfd.png",
        requesterId: "ABCEEFGCDF",
        taskId: "COMP",
        exampleIdx: 15,
        numCards: 3
    }
    """
    request = argparse.Namespace(**await request_object.json())
    # print('_fetch', request)

    config = get_config(request.taskId)

    game_instance = session.query(tables.GameInstance).filter_by(
        requester_id=request.requesterId, task_id=request.taskId).first()
    if game_instance is None:
        # print('creating new game instance')
        client_ip = get_client_ip(request_object=request_object)
        game_instance = create_game_instance(
            requester_id=request.requesterId, task_id=request.taskId, remote=client_ip,
            grammar=config.grammar, task_type=config.task_type,
            num_holdout=config.num_holdout, config=config)
        session.add(game_instance)
    session.commit()
    game_instance.example_idx
    if game_instance.num_cards is None:
        game_instance.num_cards = 2
    # print('game_instance', game_instance, 'example_idx', game_instance.example_idx)

    auto_inc = AutoInc(autoinc_every=config.autoinc_every, game_instance=game_instance)

    step_result = session.query(tables.StepResult).filter_by(
        requester_id=request.requesterId, task_id=request.taskId, example_idx=game_instance.example_idx).first()
    if increment_idx:
        if step_result.status == 'DONE':
            game_instance.example_idx += 1
            if game_instance.example_idx >= game_instance.num_steps:
                return handle_completion(game_instance=game_instance)
            step_result = None
            auto_inc(game_instance.example_idx)
        else:
            increment_idx = False
    if step_result is None:
        step_result = create_new_step_result(
            config=config, game_instance=game_instance)
        session.add(step_result)
    session.commit()

    response = {
        'messageType': 'example',
        'taskId': request.taskId,
        'requesterId': request.requesterId,
        'exampleIdx': game_instance.example_idx,
        'pictureUrl': step_result.image_path,
        'score': game_instance.score,
        'totalSteps': game_instance.num_steps,
        'maxCards': game_instance.max_cards,
        'numCards': game_instance.num_cards,
        'isHoldout': step_result.is_holdout,
        'cents_per_ten': game_instance.cents_per_ten
    }
    return web.json_response(response)


async def fetch_task(request):
    return await _fetch(request)


async def fetch_next(request):
    return await _fetch(request, increment_idx=True)


async def fetch_training_example(request_object):
    request = argparse.Namespace(**await request_object.json())
    # print('fetch_training_example', request)

    game_instance = session.query(tables.GameInstance).filter_by(
        requester_id=request.requesterId, task_id=request.taskId).first()
    if game_instance is None:
        print('game instance None => exiting')
        return web.json_response({'messageType': 'error', 'error': 'game instance not found'})

    config = get_config(task_id=request.taskId)

    task_creator = TaskCreator(
        task_type=config.task_type, seed=game_instance.seed,
        num_examples=num_examples_per_game, grammar=config.grammar)

    meaning_idx = random.randint(0, game_instance.num_cards - 1)
    # print('num_cards', game_instance.num_cards, 'meaning_idx', meaning_idx)
    ex = task_creator.create_example(idx=meaning_idx)
    # print('ex', ex)
    print(request.taskId)

    response = {
        'messageType': 'example',
        'taskId': request.taskId,
        'requesterId': request.requesterId,
        'pictureUrl': ex['filepath'],
        'utt': ex['expected'],
    }
    return web.json_response(response)


async def send_feedback(request_object):
    request = argparse.Namespace(**await request_object.json())
    print('send_feedback', request.feedback)
    game_instance = session.query(tables.GameInstance).filter_by(
        requester_id=request.requesterId, task_id=request.taskId).first()
    if game_instance is None:
        print('game instance None => exiting')
        return web.json_response({'messageType': 'error', 'error': 'game instance not found'})
    if game_instance.feedback is None:
        game_instance.feedback = ''
    game_instance.feedback = game_instance.feedback + request.feedback
    session.commit()

    response = {
        'messageType': 'receivedFeedback',
        'taskId': request.taskId,
        'requesterId': request.requesterId,
    }
    return web.json_response(response)


def handle_correct(request, step_result, game_instance):
    return_score = 1
    if step_result.status == 'TODO':
        step_result.status = 'DONE'
        step_result.score = step_result.num_cards - 1
        step_result.player_utt = request.code
        step_result.finish_datetime = datetime_utils.datetime_to_str(datetime.datetime.now())
        game_instance.score += step_result.score
        if step_result.is_holdout:
            game_instance.num_holdout_correct += 1
        result_text = f'Yes! You get {step_result.score} points!'
    else:
        result_text = 'Yes! But you already submitted for this example :)'
    return result_text, return_score


def handle_wrong(step_result, request):
    result_text = 'No. The correct code is: ' + step_result.expected_utt
    return_score = 0
    if step_result.status == 'TODO':
        step_result.status = 'DONE'
        step_result.score = 0
        step_result.player_utt = request.code
        step_result.finish_datetime = datetime_utils.datetime_to_str(datetime.datetime.now())
    return result_text, return_score


async def evaluate(request_object):
    """
    request has keys requesterId, taskId, exampleIdx, code
    response has keys requesterId, taskId, exampleIdx, resultText, score
    """
    request = argparse.Namespace(**await request_object.json())
    # print('evaluate', request)

    game_instance = session.query(tables.GameInstance).filter_by(
        requester_id=request.requesterId, task_id=request.taskId).first()
    # print('game_instance', game_instance)

    step_result = session.query(tables.StepResult).filter_by(
        requester_id=request.requesterId, task_id=request.taskId, example_idx=game_instance.example_idx).first()

    if game_instance.score is None:
        game_instance.score = 0

    if step_result.expected_utt == request.code:
        result_str = '=='
        result_text, return_score = handle_correct(
            game_instance=game_instance, request=request, step_result=step_result)
    else:
        result_str = '!='
        result_text, return_score = handle_wrong(step_result=step_result, request=request)
    session.commit()
    print(
        request.taskId, request.code, result_str, step_result.expected_utt,
        'ex=' + str(game_instance.example_idx), 'score=' + str(game_instance.score),
        'num_ho=' + str(game_instance.num_holdout_correct)
    )

    response = {
        'requesterId': request.requesterId,
        'taskId': request.taskId,
        'exampleCorrect': return_score,
        'score': game_instance.score,
        'resultText': result_text
    }
    return web.json_response(response)


async def add_card(request_object):
    request = argparse.Namespace(**await request_object.json())
    print('add_card', request)
    game_instance = session.query(tables.GameInstance).filter_by(
        requester_id=request.requesterId, task_id=request.taskId).first()
    print('game_instance', game_instance)
    if game_instance.num_cards < game_instance.max_cards - game_instance.num_holdout:
        game_instance.num_cards += 1
        session.commit()
    return web.json_response({
        'numCards': game_instance.num_cards
    })


async def remove_card(request_object):
    request = argparse.Namespace(**await request_object.json())
    print('remove_card', request)
    game_instance = session.query(tables.GameInstance).filter_by(
        requester_id=request.requesterId, task_id=request.taskId).first()
    print('game_instance', game_instance)
    if game_instance.num_cards > 2:
        game_instance.num_cards -= 1
        session.commit()
    return web.json_response({
        'numCards': game_instance.num_cards
    })


def get_client_ip(request_object):
    return request_object.headers.get('X-Real-IP', request_object.remote)


def diag(request):
    print('request.remote', request.remote, 'host', request.host)
    print('peername', request.transport.get_extra_info('peername'))
    print('headers', request.headers)
    print('client ip', request.headers.get('X-Real-IP', 'not found'))
    print('client ip', get_client_ip(request))
    response = {}
    return web.json_response(response)


if not os.path.isdir('data'):
    os.makedirs('data')

engine = sqlalchemy.create_engine('sqlite:///data/turk.db', echo=False)
print('engine', engine)
Session = sessionmaker(bind=engine)
print('Session', Session)
session = Session()
tables.create_tables(engine)

r = np.random.RandomState()

app = web.Application()
# await remotes_setup(app)
cors = aiohttp_cors.setup(app, defaults={
    '*': ResourceOptions(allow_credentials=False, allow_headers=['content-type'])
})
app.add_routes([
    web.static('/html/img', 'html/img'),
    web.static('/web', 'mll/turk/turk-web/build', show_index=False),
    web.static('/favicon', 'mll/turk/turk-web/public/favicon/', show_index=False)
])
cors.add(app.router.add_route('POST', r'/api/v1/fetch_task', fetch_task))
cors.add(app.router.add_route('POST', r'/api/v1/fetch_next', fetch_next))
cors.add(app.router.add_route('POST', r'/api/v1/fetch_training_example', fetch_training_example))
cors.add(app.router.add_route('POST', r'/api/v1/evaluate', evaluate))
cors.add(app.router.add_route('POST', r'/api/v1/add_card', add_card))
cors.add(app.router.add_route('POST', r'/api/v1/remove_card', remove_card))
cors.add(app.router.add_route('POST', r'/api/v1/send_feedback', send_feedback))
cors.add(app.router.add_route('GET', r'/api/v1/diag', diag))


if __name__ == '__main__':
    web.run_app(app)
