# coding: utf-8
import numpy, csv, bz2
import yaml

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-c', dest='conffile', required=True)
parser.add_argument('-i', dest='infile', required=True)
parser.add_argument('-o', dest='outfile', required=True)
args = parser.parse_args()

with open(args.conffile) as conffile:
    config = yaml.load(conffile)
    
with bz2.BZ2File(args.infile) as infile:
    def lines():
        readlines = 1
        while readlines > 0:
            liness = infile.readlines(5 * 1024 * 1024)
            readlines = len(liness)
            for line in liness:
                yield line
    reader = csv.reader(lines())
    data = numpy.array([map(float, line) for line in reader])
    
truths = data[:,0]
bmus = data[:,-config['network']['dim'][0]:].argmax(axis=1)
mapping = numpy.array([numpy.median(truths[bmus == i]) for i in range(config['network']['dim'][0])])
numpy.save(args.outfile, mapping)
