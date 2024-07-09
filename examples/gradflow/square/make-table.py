#/ /usr/bin/env python3

import numpy as np
import os

folderNames = ['run1', 'run2', 'run3', 'run4']
fileNames = ['run1', 'run2', 'run3', 'run4']
tableFileName = 'square-table.tex'
compile_tex = True

nruns = len(folderNames)

try:
    os.mkdir('tex')
except FileExistsError:
    pass

phi_tol = 1e-3


########################## GET DATA #############################
fvec = np.zeros(nruns)
gvec = np.zeros(nruns)
callvec = np.zeros(nruns, dtype=int)
timevec = np.zeros(nruns)
phivec = np.zeros(nruns)

for ii in range(nruns):
    data = np.load(folderNames[ii]+'/'+fileNames[ii]+'.npz', allow_pickle=True)
    t = data['t']
    x = data['xhist']
    y = data['yhist']
    k = data['khist']
    c = data['chist']
    f = data['fhist']
    g = data['ghist']
    vertices = data['vertices']
    phinormvec = data['phinormvec']

    monitor_tvec = data['monitor_tvec']
    monitor_timetotvec = data['monitor_timetotvec']
    monitor_timeCGvec = data['monitor_timeCGvec']
    monitor_CGitervec = data['monitor_CGitervec']

    fvec[ii] = f[-1]
    gvec[ii] = g[-1]
    callvec[ii] = len(monitor_timetotvec)
    timevec[ii] = np.sum(monitor_timetotvec)
    phivec[ii] = phinormvec[-1]

######################### WRITE FILE ###########################
with open('tex/'+tableFileName, 'w') as f:
    f.write(r'\begin{table}[]' + '\n')
    f.write(r'\begin{tabular}{lccccc}' + '\n')
    f.write(r'& $f_{end}$ & $\|g\|_{end}$ & $\|\bm{\Psi}\|_{end}$ & ncalls & tot Tcalls (s) \\')
    f.write(r'\cline{2-6}\noalign{\vspace{0.25ex}}' + '\n')
    for ii in range(nruns):
        f.write(folderNames[ii] + ' & ')
        f.write(r'\sisetup{round-mode = places, round-precision = 3}' + '\n')
        f.write(r'\num{'+str(fvec[ii])+r'}' + ' & ')
        f.write(r'\sisetup{round-mode = places, round-precision = 3}' + '\n')
        f.write(r'\num{'+str(gvec[ii])+r'}' + ' & ')
        f.write(r'\sisetup{round-mode = places, round-precision = 4}' + '\n')
        # Mark results not fulfilling the tolerance
        if phivec[ii]>phi_tol:
            f.write(r'{\color{red}')
        f.write(r'\num{'+str(phivec[ii])+r'}')
        if phivec[ii]>phi_tol:
            f.write(r'}')
        f.write(' & ' + str(callvec[ii]) + ' & ')
        f.write(r'\sisetup{round-mode = places, round-precision = 0}' + '\n')
        f.write(r'\num{'+str(timevec[ii])+r'}' + r'\\' +  '\n')
    f.write(r'\end{tabular}' + '\n')
    f.write(r'\end{table}' + '\n')


################## BUILD TEST FILE AND COMPILE ################
if compile_tex:
    with open('tex/test.tex', 'w') as f:
        f.write(r'\documentclass{article}' + '\n')
        f.write(r'\usepackage{siunitx}' + '\n')
        f.write(r'\usepackage{bm}' + '\n')
        f.write(r'\begin{document}' + '\n')
        f.write(r'\input{' + tableFileName + r'}' + '\n')
        f.write(r'\end{document}')
    os.system('pdflatex -output-directory=tex tex/test.tex')
    # Remove tex build files
    os.system('rm tex/*.log tex/*.aux')
