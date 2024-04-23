import os
import re

import easygui as gui

in_elc = gui.fileopenbox()
out_elc = gui.filesavebox()

if __name__ == '__main__':
    lines = []
    with open(in_elc) as f:
        lines = f.readlines()

    positions = [int(s) for s in re.findall(r'\d+', lines[0])][0]
    lastElectrodePos = positions + 3
    labelsPos = lastElectrodePos + 1
    for i in range(3, lastElectrodePos):
        lines[i] = lines[i].split(":")[1][1:]

    lines[labelsPos] = lines[labelsPos].replace('\t', '\n')

    lines = lines[:labelsPos+1]

    print(lines)

    with open(out_elc, 'a') as f:
        f.writelines(lines)
