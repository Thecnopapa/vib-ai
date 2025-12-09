import os, sys, json, xmltodict



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




