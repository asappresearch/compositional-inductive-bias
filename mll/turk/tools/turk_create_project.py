"""
e.g. see https://blog.mturk.com/tutorial-using-the-mturk-requester-website-together-with-python-and-boto-4a7ef0264b7e
"""
import argparse
from ruamel import yaml

import turk_lib

# <?xml version="1.0" encoding="UTF-8"?>

question_html_value = """
<ExternalQuestion
  xmlns="http://mechanicalturk.amazonaws.com/AWSMechanicalTurkDataSchemas/2006-07-14/ExternalQuestion.xsd">
  <ExternalURL>WEBSERVICEURL</ExternalURL>
  <FrameHeight>0</FrameHeight>
</ExternalQuestion>
"""


def run(args):
    turk = turk_lib.get_turk(prod=args.prod)

    # html_question = HTMLQuestion(question_html_value, 500)

    response = turk.create_hit(
        Question=question_html_value.replace('WEBSERVICEURL', args.external_url),
        MaxAssignments=1,
        Title="Secret Spy Codes",
        Description='Use secret codes to communicate industrial blueprints',
        Keywords='game,language',
        # Duration=120,
        Reward=str(args.reward),
        LifetimeInSeconds=int(args.lifetime_hours * 3600),
        AssignmentDurationInSeconds=300
    )
    print('')
    with open('/tmp/foo.yaml', 'w') as f:
        yaml.dump(response, f, default_flow_style=False)
    with open('/tmp/foo.yaml') as f:
        print(f.read())
    # print(yaml.dumps(response))

    # The response included several fields that will be helpful later
    # hit_type_id = response[0].HITTypeId
    # hit_id = response[0].HITId
    hit_id = response['HIT']['HITId']
    hit_type_id = response['HIT']['HITTypeId']
    print("Your HIT has been created. You can see it at this link:")
    print("https://workersandbox.mturk.com/mturk/preview?groupId={}".format(hit_type_id))
    print("Your HIT ID is: {}".format(hit_id))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prod', action='store_true')
    parser.add_argument('--lifetime-hours', type=float, default=1)
    parser.add_argument('--reward', type=float, default=0.0)
    parser.add_argument('--external-url', type=str, required=True, help='url of web service form')
    args = parser.parse_args()
    run(args)
