import json, sys

def combine(f1, f2, out):
    with open(f1) as f:
        j1 = json.load(f)

    with open(f2) as f:
        j2 = json.load(f)

    for k in j2:
        j1[k] = j2[k]

    with open(out, 'w') as f:
        json.dump(j1, f, indent=4)

def combineMultiple(fs, out):
    js = []
    for f_in in fs:
        with open(f_in) as f:
            js.append(json.load(f))

    j0 = js[0]
    for j in js[1:]:
        for k in j:
            j0[k] = j[k]

    with open(out, 'w') as f:
        json.dump(j0, f, indent=4)

if __name__ == "__main__":
    # experiment = input("Experiment (movielens/bus_breakdown/both):")
    
    if len(sys.argv) > 1:
        experiment = sys.argv[1]
        if experiment not in ["movielens", "bus_breakdown"]:
            experiments = ["movielens", "bus_breakdown"]
        else:
            experiments = [experiment]
    else:
        experiments = ["movielens", "bus_breakdown"]

    plates = {"movielens": "N20_M450", "bus_breakdown": "M3_J3_I30"}

    for e in experiments:
        for metric in ["elbo", "MSE", "p_ll", "variance"]:
            combine(f"{e}/results/{e}_{metric}_{plates[e]}.json",
                    f"{e}/results/vi_{e}_{metric}_{plates[e]}.json",
                    f"{e}/results/ALL_{e}_{metric}_{plates[e]}.json")