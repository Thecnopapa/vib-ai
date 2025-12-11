import os, sys, json, xmltodict

import bioiain as bi
import subprocess


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
    molecules = {}
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
                }
                if molecule["chain_id"] not in molecules:
                    molecules[molecule["chain_id"]] = {
                        "id": molecule["id"],
                        "chain_id": molecule["chain_id"],
                        "class": molecule["class"],
                        "residues": {}
                    }
                m = molecules[molecule["chain_id"]]
                for n, residue in enumerate(molecule["residues"]["residue"]):
                    if type(residue) == str:
                        continue
                    print(residue)
                    if residue["ser_no"] not in m["residues"]:
                        m["residues"][residue["ser_no"]] = {
                            "ser_no": residue["ser_no"],
                            "name": residue["name"],
                            "seq_num": residue["seq_num"],
                            "label_seq_num": residue["label_seq_num"],
                            "interactions": {}
                            }
                    solv_en = float(residue["solv_en"])
                    if  solv_en != 0:
                        m["residues"][residue["ser_no"]]["interactions"][interface["id"]] = {
                            "asa": float(residue["asa"]),
                            "bsa": float(residue["bsa"]),
                            "solv_en": solv_en,
                        }
        data["interfaces"] = interfaces
        data["molecules"] = molecules

    assemblies = {}
    with open(assemblies_path) as f:
        xml = xmltodict.parse(f.read())["pisa_results"]
        if xml["status"] != "Ok":
            raise Exception(f"Assembly XML error: {xml}")
        data["pisa_id"] = xml["name"]
        data["multimeric_state"] = xml["multimeric_state"]
        data["assessment"] = xml["assessment"]
        data["n_assembly_groups"] = xml["total_asm"]
        xml = xml["asm_set"]
        if type(xml) != list:
            xml = [xml]
        for assembly_group in xml:
            assembly_serial = assembly_group["ser_no"]
            print(">>>> serial group:", assembly_serial)
            #print(assembly_group["assembly"]["serial_no"])
            if type(assembly_group["assembly"]) == dict:
                asss = [assembly_group["assembly"]]
            elif type(assembly_group["assembly"]) == list:
                asss = assembly_group["assembly"]
            else:
                raise Exception(f"Assembly group XML error: {assembly_group['assembly']}")
            #print_children(assembly_group["assembly"])
            for assembly in asss:
                print(">>>> assembly:", assembly["serial_no"])
                print_children(assembly)
                assemblies[assembly["serial_no"]] = {
                    "info": {k:v for k,v in assembly.items() if type(v) not in [list, dict]},
                    "assembly_group": assembly_serial,
                    "interfaces": {}
                }
                a = assemblies[assembly["serial_no"]]
                a["info"]["n_interfaces"] = int(assembly["interfaces"]["n_interfaces"])
                #print(json.dumps(assemblies[assembly["serial_no"]], indent=4))
                if a["info"]["n_interfaces"] > 0:
                    #print_children(assembly["interfaces"]["interface"])
                    inters = assembly["interfaces"]["interface"]
                    if type(inters) != list:
                        inters = [inters]
                    for interface in inters:
                        #a["interfaces"][interface["id"]] = {"info": {k:v for k,v in assembly.items() if type(v) not in [list, dict]}}
                        a["interfaces"][interface["id"]] = {"id": interface["id"], "dissociates": interface["dissociates"]}

        data["assemblies"] = assemblies
    print(json.dumps(data["assemblies"], indent=4))
    json.dump(data, open("out.json", "w"), indent=4)
    #print_children(data)




def _pisa_to_xml(pisa_id, folder="pisa_raw", interfaces=True, assemblies=True, pisa_command="pisa"):
    cmd = [
        pisa_command,
        pisa_id,
        "-xml",
    ]
    if interfaces:
        cmd_i = cmd + ["interfaces", ">", f"{folder}/{pisa_id}.interfaces.xml"]
        subprocess.run(cmd_i)
    if assemblies:
        cmd_a = cmd + ["assemblies", ">", f"{folder}/{pisa_id}.assemblies.xml"]
        subprocess.run(cmd_a)
    return None, None

def run_pisa(filepath,  pisa_id="temp", pisa_command="pisa", xml_folder="pisa_raw", interfaces=True, assemblies=True):
    try:
        ccp4_path = os.environ["CCP4"]
    except KeyError:
        bi.log("error", "CCP4 not enabled")
        ccp4_path = None
    if ccp4_path is None:
        return None
    print("CCP4 path:", ccp4_path)
    filepath = os.path.abspath(filepath)
    cmd = [
        pisa_command,
        pisa_id,
        "-analyse",
        filepath
    ]
    try:
        subprocess.run(cmd, check=True)
    except:
        return None
    print("PISA DONE")

    if interfaces or assemblies:
        int_file, ass_fie = _pisa_to_xml(pisa_id, folder=xml_folder, interfaces=interfaces, assemblies=assemblies)
        return pisa_id, int_file, ass_fie
    else:
        return pisa_id









session = run_pisa("1M2Z.cif")
parse_pisa(session)






