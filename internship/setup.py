import json, os, sys
from bioiain import log

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
print(json.dumps(config, indent=2))