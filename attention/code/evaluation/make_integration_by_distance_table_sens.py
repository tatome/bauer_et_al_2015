import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-i', dest='infiles', nargs='+', required=True)
parser.add_argument('-o', dest='outfile', required=True)
args = parser.parse_args()

data = [np.load(f) for f in args.infiles]

# last data file is from the simulation with the base configuration.
base_data = data[-1]

def config(data):
    return data['config'].item()

def amps(d):
    vis = config(d)['modalities'][0]['activation']['amplitude']
    aud = config(d)['modalities'][1]['activation']['amplitude']
    return vis,aud

def akaike(which):
    def special_akaike(data):
        data = data[which]
        distances = data[0]
        akaike_int = data[3]
        akaike_sel = data[4]
        mm = (np.where((distances > .1) & (akaike_int > akaike_sel))[0])
        if len(mm) > 0:
            crit = np.where((distances > .1) & (akaike_int > akaike_sel))[0][0]
        else:
            crit = len(distances) - 1
        return distances[crit]
    return special_akaike

def mean(which):
    def special_mean(data):
        return data[which].item()
    return special_mean

def std(which):
    def special_std(data):
        data = data[which]
        stds = data[2]
        return stds.min()
    return special_std

columns = [
    ['$a_{\mathit{Va}}$',      akaike('vis_data'),     "%.3f"],
    ['$a_{\mathit{vA}}$',      akaike('aud_data'),     "%.3f"],
    ['$a_{\mathit{VA}}$',      akaike('both_data'),    "%.3f"],
    ['$a_{\mathit{va}}$',      akaike('none_data'),    "%.3f"],
    ['$\\mu_{\mathit{Va}}$',    mean('vis_mean'),       "%.3f"],
    ['$\\mu_{\mathit{vA}}$',    mean('aud_mean'),       "%.3f"],
    ['$\\mu_{\mathit{VA}}$',    mean('both_mean'),      "%.3f"],
    ['$\\mu_{\mathit{va}}$',    mean('none_mean'),      "%.3f"],
#    ['$\\sigma_{\mathit{Va}}$', std('vis_data'),        "%.3f"],
#    ['$\\sigma_{\mathit{vA}}$', std('aud_data'),        "%.3f"],
#    ['$\\sigma_{\mathit{VA}}$', std('both_data'),       "%.3f"],
#    ['$\\sigma_{\mathit{va}}$', std('none_data'),       "%.3f"],
]
header = " & ".join(["$g_v,g_a$"] + [c[0] for c in columns]) + "\\\\\n"
col_desc = "r|" + "r" * len(columns)

data = sorted(data, key=amps)
with open(args.outfile, 'w') as outfile:
    outfile.write("""{\\small%\n""")
    outfile.write("\\begin{tabular}{" + col_desc + "}\n")
    outfile.write(header)
    for d in data:
        line    = [amps(d)] +     [c[1](d) for c in columns]
        formats = ["%.1f,%.1f"] + [c[2]    for c in columns]

        if amps(d) == amps(base_data):
            template = "$\\mathbf{%s}$"
        else:
            template = "$%s$"
        line = [template % (f % e) for f,e in zip(formats,line)]
        outfile.write(" & ".join(line) + "\\\\\n")

    outfile.write("\\end{tabular}\n}")
