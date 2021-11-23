import argparse

import turk_lib


def run(args):
    turk = turk_lib.get_turk(prod=args.prod)

    hits = turk.list_hits(MaxResults=100)['HITs']
    print('len(hits)', len(hits))
    for item in hits:
        hit_id = item['HITId']
        # print('HITId:', hit_id)

        hit = turk.get_hit(HITId=hit_id)['HIT']

        """
        HIT:
        AssignmentDurationInSeconds: 300
        AutoApprovalDelayInSeconds: 2592000
        CreationTime: 2021-11-13 07:51:51-05:00
        Description: Use secret codes to communicate industrial blueprints
        Expiration: 2021-11-14 07:51:51-05:00
        HITGroupId: 3LUKZ8GFY2HIPIHYBUK68Z1NAP1WGN
        HITId: 3OCZWXS7ZO83TRH48FK3L860S025LV
        HITReviewStatus: NotReviewed
        HITStatus: Assignable
        HITTypeId: 32HMK6WX8ZM9LXL02OQQBROS2B73KV
        Keywords: game,language
        MaxAssignments: 1
        NumberOfAssignmentsAvailable: 1
        NumberOfAssignmentsCompleted: 0
        NumberOfAssignmentsPending: 0
        QualificationRequirements: []
        Question: "..."
        Reward: '0.05'
        Title: Secrete Spy Codes
        """
        status = hit['HITStatus']
        # hit_type_id = hit['HITTypeId']
        title = hit['Title']
        # description = hit['Description']
        available = hit['NumberOfAssignmentsAvailable']
        pending = hit['NumberOfAssignmentsPending']
        completed = hit['NumberOfAssignmentsCompleted']
        # total = available + pending + completed
        reviewed = hit['HITReviewStatus']
        created = str(hit['CreationTime'])[:16]
        group_id = hit['HITGroupId']
        print(f'{created} {hit_id[:4]} {status} {reviewed} {available}/{pending}/{completed} {title.ljust(25)}')
        print("https://workersandbox.mturk.com/mturk/preview?groupId={}".format(group_id))
        print('')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prod', action='store_true')
    args = parser.parse_args()
    run(args)
