
# Common Libraries
import numpy as np

# Functions
def quick_sort(itr, cmp):
    if len(itr) <= 1:
        return itr
    else:
        pivot = itr[0]

        left_pivot = [e for e in itr[1:] if cmp(e,pivot) == True]
        right_pivot = [e for e in itr[1:] if cmp(e,pivot) == False]
        
        return quick_sort(left_pivot, cmp) + [pivot] + quick_sort(right_pivot, cmp)

def sort_usingRef(targets, refs, cmp):
    """
    Sort list using reference by compare method
    
    :param targets: population to be sorted(list)
    :param refs: reference population to sort(list)
    :param cmp: compare method(function) ex) lambda a, b: a < b
    
    return (list)
    """
    # Validation check - all element must be unique
    assert len(np.unique(refs)) == len(refs), "Each element must be unique"
    
    # Sort reference population
    sorted_refs = quick_sort(refs, cmp)
    
    # Mapping target and reference using index
    ref_indexes = [refs.index(ref) for ref in sorted_refs]
    return [targets[index] for index in ref_indexes]
