from typing import List


def ref_range_to_refs(ref_range: str) -> List[str]:
    """
    Given a ref range like eg ibr152-ibr155
    returns a list of the individual refs, eg
    [ibr152,ibr153,ibr154,ibr155]

    assumptions:
    - always comprises alphabetic then number
    - alphabetic bit always constant
    - number always same number of digits
    - only two refs in range, separated by '-'
    """
    prefix = ''
    prefix_length = 0
    left, right = ref_range.split('-')
    assert len(left) == len(right)
    while prefix_length < len(left) and left[prefix_length] == right[prefix_length]:
        prefix_length += 1
    prefix = left[:prefix_length]
    start = int(left[prefix_length:])
    numeric_length = len(left) - prefix_length
    end = int(right[prefix_length:])
    refs = [prefix + str(idx).rjust(numeric_length, '0') for idx in range(start, end + 1)]
    return refs
