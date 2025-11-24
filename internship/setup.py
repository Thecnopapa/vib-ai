import json, os, sys


def import_bi():
    global bi
    global bioiain
    local_bi = "local-bi" in sys.argv
    try:
        if local_bi:
            raise ImportError("bioiain")
        import bioiain
        import bioiain as bi

    except:
        try:
            import importlib
            sys.path.append("/home/iain/projects/bioiain")
            import src.bioiain as bi
            bioiain = bi
        except:
            raise ImportError("bioiain")




def configuration():
    from bioiain import log
    global config

    config = json.load(open("config.default.json"))
    if os.path.exists("config.custom.json"):
        config = config | json.load(open("config.custom.json"))

    if "--config" in sys.argv:
        cj_index = sys.argv.index("--config") + 1
        try:
            config = config | json.load(open(sys.argv[cj_index]))
        except:
            log("error","config file could not be loaded: {}".format(os.path.abspath(sys.argv[cj_index])))
            exit()
    #print(json.dumps(config, indent=2))


def init():
    global bi
    global bioiain
    global config
    bi = None
    bioiain = None
    config = None
    import_bi()
    configuration()