from ruamel import yaml
import boto3


def get_credentials():
    with open('.aws_config.yaml') as f:
        aws_config = yaml.safe_load(f)['turk']
    return aws_config


def get_turk(prod: bool):
    aws_config = get_credentials()
    extra_kwargs = {}
    if not prod:
        extra_kwargs['endpoint_url'] = 'https://mturk-requester-sandbox.us-east-1.amazonaws.com'

    turk = boto3.client(
        'mturk',
        **extra_kwargs,
        **aws_config
    )

    if not prod:
        assert 'sandbox' in turk._endpoint.host

    print('')
    print(turk._endpoint.host)
    print('')
    return turk
