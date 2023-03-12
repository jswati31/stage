import importlib


def _get_model_name(model_name):
    '''
    given the model_name
    the file models.model_name.py will
    be imported
    '''
    model_filename = 'models.' + model_name.lower()
    modellib = importlib.import_module(model_filename)
    model = None
    target_model_name = model_name
    for name, cls in modellib.__dict__.items():
        if name.lower() == target_model_name.lower():
            model = cls

    if model is None:
        print('In %s.py, there should be a model with class name that matches %s in lowercase'%(model_filename,target_model_name))
    return model


def create_model(cfg):
    model = _get_model_name(cfg.model_type)
    instance = model(cfg)
    print("model [%s] was created" % (cfg.model_type))
    return instance

