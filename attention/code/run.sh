echo "Warning: This will run the full simulation which may take a long time.  Press [Ctrl] + [C] to stop." >&2
sleep 20
cd `/usr/bin/dirname $0`
echo `hostname`":"`pwd`
/usr/bin/python attention.py -o ../simulation/ -l ../simulation/simulation.log --processes -1
