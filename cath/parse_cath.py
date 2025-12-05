import os, json

from bioiain.utilities import str_to_list_with_literals


def parse_cath(code, chain, domain=None, cath_folder="/home/iain/downloads/orengoftp.biochem.ucl.ac.uk/cath/releases/latest-release/cath-classification-data/"):

    dom_list_file = os.path.join(cath_folder, "cath-domain-list.txt")

    with open(dom_list_file) as f:
        for line in f:
            if line[0] == "#":
                continue


            comps = str_to_list_with_literals(line)
            c = comps[0][:4]
            ch = comps[0][4:5]
            dom = comps[0][5:7]

            if code.upper() != c.upper():
                continue

            if chain.upper() != ch.upper():
                continue

            if domain != None:
                if int(domain) != int(dom):
                    continue

            info = dict(
                dom_name = comps[0],
                class_number = comps[1],
                arch_number = comps[2],
                top_number = comps[3],
                hom_fam_number = comps[4],
                s35 = comps[5],
                s60 = comps[6],
                s95 = comps[7],
                s100 = comps[8],
                s100_count = comps[9],
                dom_len = comps[10],
                res = comps[11]
            )
            print(json.dumps(info))

parse_cath("1M2Z", "A")






