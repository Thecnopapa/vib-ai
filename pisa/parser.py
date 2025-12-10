import os, sys, json, xmltodict




def test():
    with open("assemblies.xml") as f:
        xml = xmltodict.parse(f.read())
        print(json.dumps(xml, indent=4))

    with open("interactions.xml") as f:
         xml = xmltodict.parse(f.read())
         #print(json.dumps(xml, indent=4))
         print("strings:")
         [print(k, v) for k, v in xml["pdb_entry"]["interface"][0]["molecule"][0]["residues"]["residue"][0].items() if type(v) == str]
         print("other:")
         [print(k,type(v), len(v)) for k, v in xml["pdb_entry"]["interface"][0]["molecule"][0]["residues"]["residue"][0].items() if type(v) != str and v is not None]
         #print(xml["pdb_entry"]["interface"][0].keys())

def print_children(d):
    if type(d) == list:
        d = d[0]
        print("(list)[0]")
    print("strings:")
    [print(k, v) for k, v in d.items() if type(v) == str]
    print("other:")
    [print(k, type(v), len(v)) for k, v in d.items() if type(v) != str and v is not None]

def parse_pisa(pisa_id, folder="pisa_raw", name=None):
    interactions_path = os.path.join(folder, f"{pisa_id}.interactions.xml")
    assemblies_path = os.path.join(folder, f"{pisa_id}.assemblies.xml")

    if name is None:
        name = pisa_id
    assert os.path.isfile(assemblies_path)
    assert os.path.isfile(interactions_path)
    data = {}
    interfaces = {}
    with open(interactions_path) as f:
        xml = xmltodict.parse(f.read())["pdb_entry"]
        if xml["status"] != "Ok":
            raise Exception(f"Interaction XML error: {xml}")
        data["pdb_code"] = xml["pdb_code"]
        data["n_interfaces"] = xml["n_interfaces"]
        xml = xml["interface"]
        for interface in xml:
            interfaces[interface["id"]] = {
                "info": {k:v for k,v in interface.items() if type(v) not in [list, dict]},
                "molecules": {}
                }
            i = interfaces[interface["id"]]
            for molecule in interface["molecule"]:
                i["molecules"][molecule["id"]] = {
                    "info": {k:v for k,v in molecule.items() if type(v) not in [list, dict]},
                    "residues": {}
                }
                m = i["molecules"][molecule["id"]]
                for n, residue in enumerate(molecule["residues"]["residue"]):
                    if type(residue) == str:
                        continue
                    m["residues"][n] = {k:v for k,v in residue.items() if type(v) not in [list, dict]}
        data["interfaces"] = interfaces

    assemblies = {}
    with open(assemblies_path) as f:
        xml = xmltodict.parse(f.read())["pisa_results"]
        if xml["status"] != "Ok":
            raise Exception(f"Interaction XML error: {xml}")
        data["pisa_id"] = xml["name"]
        data["multimeric_state"] = xml["multimeric_state"]
        data["assessment"] = xml["assessment"]
        data["n_assemblies"] = xml["total_asm"]


        print_children(xml)







    #print(json.dumps(data, indent=4))
    #print_children(data)



parse_pisa("test")






