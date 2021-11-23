import requests
import argparse


def run(args):
    request = {
        'requesterId': args.requester_id,
        'taskId': args.task_id
    }
    res = requests.post(
        f'http://{args.host}:{args.port}/api/v1/fetch_task', json=request
    )
    if res.ok:
        print(res.json())
    else:
        print('error')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--requester-id', type=str, default='ABC')
    parser.add_argument('--task-id', type=str, default='COMP')
    parser.add_argument('--host', type=str, default='localhost')
    parser.add_argument('--port', type=int, default=8000)
    args = parser.parse_args()
    run(args)
