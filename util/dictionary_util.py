
def search_dict(dictionary, keywords):
    """
    Search keywords over values in dictionary 
    
    This function iterates dictionary using keys for searching keyword in value
    
    return: searched dictionary
    """
    keys = []
    infos = []
    for key in dictionary:
        is_not_matching = False
        for keyword in keywords:
            if keyword not in dictionary[key]:
                is_not_matching = True

        if is_not_matching == False:
            keys.append(key)
            infos.append(dictionary[key])
    return dict(zip(keys, infos))
