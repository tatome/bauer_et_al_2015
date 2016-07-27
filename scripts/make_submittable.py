import StringIO
import os.path
import shutil
import re

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-m', dest='mainfile', required=True)
parser.add_argument('-o', dest='outfile', required=True)
args = parser.parse_args()

# Collate into one file
with open(args.mainfile) as mainfile:
    text = mainfile.read()
input_re = r'.*\input{([^}]*)}.*'
matcher = re.compile(input_re)
found_something = True
while found_something:
    found_something = False
    outfile = StringIO.StringIO()
    infile = StringIO.StringIO(text)
    for line in infile:
        m = matcher.match(line)
        if m is None:
            outfile.write(line)
        else:
            if m.group(1) not in ['tikzstuff', 'drafting']:
                print("including file %s" % m.group(1) + '.tex')
                # assuming max. one input per line, filenames w/out extensions.
                with open(m.group(1) + '.tex') as included:
                    contents = included.read()
                newline = line.replace(m.group(0), contents)
                outfile.write(newline)
                found_something = True
    text = outfile.getvalue()

# Collect tikz picture names.
image_log_entry_re = r'^=+ Image \'([^\s]*)\' is up-to-date. =+$'
matcher = re.compile(image_log_entry_re)
image_filenames = []
logfilename = args.mainfile[:-4] + '.log'
with open(logfilename) as logfile:
    for line in logfile:
        m = matcher.match(line)
        if m:
            print("Found image: %s" % m.group(1))
            image_filenames.append(m.group(1) + '.pdf')

image_filename_replacements = []
for cntr,image_filename in enumerate(image_filenames):
    prefix = 'figure-%02d-' % (cntr + 1)
    newname = prefix + os.path.basename(image_filename)
    image_filename_replacements.append(newname)

image_filename_replacements = dict(zip(image_filenames, image_filename_replacements))

for oldname in image_filename_replacements:
    shutil.copy(oldname, os.path.join('submittable', image_filename_replacements[oldname]))


# Replace tikz images by pdfs.
includetikz_re = r'.*\\includetikz{([^}]+)}.*|.*\\includepgf{([^}]+)}.*'
matcher = re.compile(includetikz_re)
infile = StringIO.StringIO(text)
outfile = StringIO.StringIO()
padding = '\\includegraphics{%s}\n'
for line in infile:
    m = matcher.match(line)
    if m is None:
        outfile.write(line)
    else:
        repl = m.group(1) if m.group(1) is not None else m.group(2)
        outfile.write(padding % image_filename_replacements[repl + '0.pdf'])
text = outfile.getvalue()

with open(args.outfile, 'wc') as outfile:
    outfile.write(text)

shutil.copy('bib.bib', 'submittable')
