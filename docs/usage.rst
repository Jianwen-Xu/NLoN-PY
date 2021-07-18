=====
Usage
=====

To use nlon-py in a project::

    import nlon_py

To use the default model::

    from nlon_py.data.build_model import loadDefaultModel, transform_data
    
    from nlon_py.model import NLoNPredict

    model = loadDefaultModel()

    text = ['This is natural language.',
            'public void NotNaturalLanguageFunction(int i, String s)']

    y = NLoNPredict(model, transform_data(text))

    print(y) # output e.g. ['NL', 'CODE']
