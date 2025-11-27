import json, os, sys, datetime
now = datetime.datetime.now()
os.makedirs("logs", exist_ok=True)
log_file = "log_{}".format(now)
log_file = log_file.replace(":", "_")
log_file = log_file.replace(".", "_")

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


    if "--setup" in sys.argv:
        try:
            setup_name = sys.argv[sys.argv.index("--setup") + 1]
        except:
            bi.log("error", "Setup name not provided")
            exit()
    else:
        setup_name = config["setups"]["default"]
    bi.log("header", "Setup name:", setup_name)
    config["setups"]["selected"] = config["setups"][setup_name]
    config["data"]["default"] = config["setups"]["selected"]["data"]
    config["labels"]["default"] = config["setups"]["selected"]["labels"]
    config["embeddings"]["default"] = config["setups"]["selected"]["embeddings"]
    config["training"]["default"] = config["setups"]["selected"]["training"]
    if log_file is not None:
        with open(f"logs/{log_file}", "a") as log:
           log.write(f"""
{now}
    
### ### Log file: {log_file}
    
### CMD

    {" ".join(sys.argv)}
    
###
""")

        if "--data" in sys.argv:
            try:
                dataset = sys.argv[sys.argv.index("--data") + 1]
            except:
                bi.log("error", "Dataset not provided")
                exit()
        else:
            dataset = config["data"]["default"]
        bi.log(1, "Using dataset:", dataset)
        config["data"]["selected"] = config["data"][dataset]




    if "--labels" in sys.argv:
        label_method = sys.argv[sys.argv.index("--labels") + 1]
    else:
        label_method = config["labels"]["default"]
    bi.log(1, "Label method:", label_method)
    config["labels"]["selected"] = config["labels"][label_method]


    if "--embedding" in sys.argv:
        embedding_method = sys.argv[sys.argv.index("--embedding") + 1]
    else:
        embedding_method = config["embeddings"]["default"]
    bi.log(1, "Embedding method:", embedding_method)
    config["embeddings"]["selected"] = config["embeddings"][embedding_method]


    if "--model" in sys.argv:
        try:
            training_setting = sys.argv[sys.argv.index("--model") + 1]
        except:
            bi.log("error", "Training settings not provided")
            exit()
    else:
        training_setting = config["training"]["default"]
    bi.log(1, "Training setting:", training_setting)
    config["training"]["selected"] = config["training"][training_setting]




def init():
    global bi
    global bioiain
    global config
    bi = None
    bioiain = None
    config = None
    import_bi()
    configuration()