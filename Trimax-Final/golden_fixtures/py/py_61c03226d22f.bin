def to_obj(cls, obj_data=None, *fields, **field_map):
    ''' Use obj_data (dict-like) to construct an object of type cls
    prioritize obj_dict when there are conficts '''
    if not fields:
        fields = obj_data.keys()
    try:
        kwargs = {field(f, field_map): obj_data[f] for f in fields if f in obj_data}
        obj = cls(**kwargs)
    except:
        getLogger().exception("Couldn't use kwargs to construct object")
        # use default constructor
        obj = cls()
        update_obj(obj_data, obj, *fields, **field_map)
    return obj