#!/usr/bin/env python
# coding: utf-8
import os, sys, math
import numpy as np
import glob
import mdtraj
import logging
import netCDF4 as nc
import warnings
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, AutoMinorLocator
import seaborn as sns
import barnaba as bb
from barnaba import definitions
from barnaba.nucleic import Nucleic


#
# SETTINGS
#

# logging
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

# plot settings
pd.options.display.max_rows = None
pd.options.display.max_columns = None
pd.options.display.precision = 1
pd.options.display.float_format = '{:.1f}'.format

params = {'legend.fontsize': 40, 
          'font.size': 40, 
          'axes.labelsize': 48,
          'axes.titlesize': 48,
          'xtick.labelsize': 40,
          'ytick.labelsize': 40,
          'savefig.dpi': 600, 
          'figure.figsize': [64, 8],
          'xtick.major.size': 10,
          'xtick.minor.size': 7,
          'ytick.major.size': 10,
          'ytick.minor.size': 7}

plt.rcParams.update(params)


#
# BACKBONE DEFINITION
#
backbone_sugar_atoms = [
    "C1'", \
    "H1'", \
    "C2'", \
    "H2'", \
    "C3'", \
    "H3'", \
    "C4'", \
    "H4'", \
    "C5'", \
    "H5'", \
    "H5''", \
    "O2'", \
    "HO2'", \
    "O3'", \
    "O4'", \
    "O5'", \
    "P", \
    "OP1", \
    "OP2", \
    "HO5'", \
    "HO3'"
]



def radian_to_degree(a):
    """
    a : list
        [trajectory frame : residue : torsion]
    """
    
    a[np.where(a<0.0)] += 2.*np.pi
    a *= 180.0/np.pi

    # same as above
    #a = a*(180./np.pi)
    #a[np.where(a<0.0)] += 360
    
    return a



def calc_bb_torsion(init_pdb, traj, rnames):
    """
    Torsion distribution
    """
    # Compare backbone angles for initial and minimized structure
    backbone_annot_dict = {}
    backbone_annot_dict["alpha"]=0
    backbone_annot_dict["beta"]=1
    backbone_annot_dict["gamma"]=2
    backbone_annot_dict["delta"]=3
    backbone_annot_dict["eps"]=4
    backbone_annot_dict["zeta"]=5
    backbone_annot_dict["chi"]=6

    cols = []
    cols += [ a for a in definitions.bb_angles ]

    # production
    angles, res = bb.backbone_angles_traj(traj)
    a = radian_to_degree(np.array(angles))       # move from -pi,pi to 0-2pi range and convert radians to deg
    bins = np.arange(0, 360, 10)

    fig = plt.figure(figsize=(24, 18))
    for k, v in backbone_annot_dict.items():
        # k : torsion name
        # v : torsion index
        ax = fig.add_subplot(4, 3, v+1)
        for i in range(len(rnames)):
            # i : residue name index
            # x : normalized frequency
            # y : degree
            y, x = np.histogram(a[:,i, v], density=True, bins=bins)
            ax.plot(0.5*(x[1:]+x[:-1]), y, label="{}-{}".format(rnames[i], i))
        if v == 0:
            plt.legend(loc="upper left", fontsize=18)
        if v == 3:
            ax.set_ylabel("Probability", fontsize=32)
        ax.set_xlabel("{} angle (deg)".format(k), fontsize=32)
        ax.set_xlim(0, 360)
        ax.set_ylim(0, 0.06)
        ax.tick_params(axis='both', labelsize=18)

    plt.tight_layout()
    plt.savefig("bb_angles.png")



def calc_sugar_pucker(init_pdb, traj, rnames):
    """
    sugar pucker
    """
    # initial structure
    init_angles, res = bb.pucker_angles(init_pdb, topology=init_pdb)

    # production
    angles, res = bb.pucker_rao_traj(traj)

    fig = plt.figure(figsize=(24, 18))
    for i in range(len(rnames)):
        ax = fig.add_subplot(1, 4, i+1, polar=True)
        #ax.plot(polar=True)

        ax.scatter(angles[:,i,0], angles[:,i,1], s=10, c=np.arange(len(angles)), cmap='Blues', label="{}-{}".format(rnames[i], i))
        ax.scatter(init_angles[:,i,0], init_angles[:,i,1], marker="X", c="orange", edgecolors="black", s=150, linewidths=0.5)
        
        p3 = np.pi/5
        ax.text(0.5*p3, 1.6, "C3'-endo", ha='center', fontsize=16, fontweight='bold')
        ax.text(1.3*p3, 1.5, "C4'-exo",  ha='center', fontsize=16)
        ax.text(2.5*p3, 1.5, "O4'-endo", ha='center', fontsize=16)
        ax.text(3.7*p3, 1.5, "C1'-exo",  ha='center', fontsize=16)
        ax.text(4.5*p3, 1.6, "C2'-endo", ha='center', fontsize=16, fontweight='bold')
        ax.text(5.5*p3, 1.5, "C3'-exo",  ha='center', fontsize=16)
        ax.text(6.5*p3, 1.5, "C4'-endo", ha='center', fontsize=16)
        ax.text(7.5*p3, 1.6, "O4'-exo",  ha='center', fontsize=16)
        ax.text(8.5*p3, 1.5, "C1'-endo", ha='center', fontsize=16)
        ax.text(9.5*p3, 1.5, "C2'-exo",  ha='center', fontsize=16)
        
        xt = np.arange(0, 2*np.pi, p3)
        ax.set_xticks(xt)
        ax.set_yticks([])
        ax.set_ylim(0, 1.2)
        ax.tick_params(axis='both', labelsize=12)
        
        plt.tight_layout()
        #plt.legend(loc="upper center")
        plt.savefig("pucker_angles.png")



def calc_rg(init_pdb, traj, rnames):
    """
    Radius of gyration
    """
    #
    # histogram
    #
    atom_indices = traj.topology.select('not (protein or water or symbol Na or symbol Cl)')
    t = traj.atom_slice(atom_indices)
    rg = mdtraj.compute_rg(t)
    rg = np.array(rg) * UNIT_NM_TO_ANGSTROMS   # convert nm to angstroms

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.set_title(PLOT_TITLE)

    # x-axis
    ax.set_xlabel(r'Rg [${\rm \AA}$]')
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    
    # y-axis
    ax.set_ylabel('Probability')
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.hist(rg, bins=20, density=True)
    plt.tight_layout()
    plt.savefig("rg_hist.png")

    #
    # time plot
    #
    x = np.arange(1, len(rg)+1) * LOGGING_FREQUENCY * STRIDE * UNIT_PS_TO_NS * UNIT_PS_TO_NS # microsecond

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.set_title(PLOT_TITLE)

    # x-axis
    #ax.set_xlabel(r'time [ns]')
    ax.set_xlabel(r'Time [${\rm \mu}$s]')
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    #ax.set_xlim([0, len(x)]) 
    ax.set_xlim([0, x.max()]) 
    
    # y-axis
    ax.set_ylabel(r'Rg [${\rm \AA}$]')
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.set_ylim([4.5, 8])

    ax.plot(x, rg, lw=1)
    plt.tight_layout()
    plt.savefig("rg.png")



def calc_rmsd(init_pdb, traj, rnames):
    """
    RMSD and eRMSD
    ---------

    RMSD:  Calculate rmsd after optimal alignment between reference and target structures. Superposition and RMSD calculations are performed using all heavy atoms. If the sequence of reference and target is different, only backbone/sugar heavy atoms are used.  
    eRMSD: Calculate ermsd between reference and target structures  
    """

    #
    # RMSD time plot
    #
    init_traj = mdtraj.load(init_pdb)
    rmsd = list(bb.functions.rmsd_traj(init_traj, traj))   
    rmsd = np.array(rmsd) * UNIT_NM_TO_ANGSTROMS

    x = np.arange(1, len(rmsd)+1) * LOGGING_FREQUENCY * STRIDE * UNIT_PS_TO_NS * UNIT_PS_TO_NS   # microsecond

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.set_title(PLOT_TITLE)

    # x-axis
    #ax.set_xlabel(r'Time [ns]')
    ax.set_xlabel(r'Time [${\rm \mu}$s]')
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    #ax.set_xlim([0, len(x)]) 
    ax.set_xlim([0, x.max()])
    
    # y-axis
    ax.set_ylabel(r'RMSD [${\rm \AA}$]')
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.set_ylim([0, 6])

    ax.plot(x, rmsd, lw=1)
    plt.tight_layout()
    plt.savefig("rmsd.png")


    #
    # eRMSD time plot
    #
    ermsd = list(bb.functions.ermsd_traj(init_traj, traj))   
    ermsd = np.array(ermsd) * UNIT_NM_TO_ANGSTROMS

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.set_title(PLOT_TITLE)

    # x-axis
    #ax.set_xlabel(r'Time [ns]')
    ax.set_xlabel(r'Time [${\rm \mu}$s]')
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    #ax.set_xlim([0, len(x)]) 
    ax.set_xlim([0, x.max()]) 

    # y-axis
    ax.set_ylabel(r'eRMSD [${\rm \AA}$]')
    ax.yaxis.set_minor_locator(AutoMinorLocator())

    ax.plot(x, ermsd, lw=1)
    plt.tight_layout()
    plt.savefig("ermsd.png")


    #
    # RMSD and eRMSD side-by-side plot
    #
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(24, 8))
    fig.suptitle(PLOT_TITLE, y=0.85)

    # xy-axis (1)
    #ax1.set_xlabel(r'Time [ns]')
    ax1.set_xlabel(r'Time [${\rm \mu}$s]')
    ax1.xaxis.set_minor_locator(AutoMinorLocator())
    #ax1.set_xlim([0, len(x)]) 
    ax1.set_xlim([0, x.max()]) 
    ax1.set_ylabel(r'RMSD [${\rm \AA}$]')
    ax1.yaxis.set_minor_locator(AutoMinorLocator())
    ax1.set_ylim([0, 6])

    # xy-axis (2)
    #ax2.set_xlabel(r'Time [ns]')
    ax2.set_xlabel(r'Time [${\rm \mu}$s]')
    ax2.xaxis.set_minor_locator(AutoMinorLocator())
    #ax2.set_xlim([0, len(x)]) 
    ax2.set_xlim([0, x.max()]) 
    ax2.set_ylabel(r'eRMSD [${\rm \AA}$]')
    ax2.yaxis.set_minor_locator(AutoMinorLocator())
    
    # plot
    ax1.plot(x, rmsd, lw=1, c='b')
    ax2.plot(x, ermsd, lw=1, c='r')

    plt.tight_layout()
    plt.savefig("rmsd_ermsd.png")



def _check_endo(angle_d, angle_p):
    """
    Define C3'-endo and C2'-endo.

    δ torsion angles is used to defined the endo states described in:
    RNA backbone: Consensus all-angle conformers and modular string nomenclature (an RNA Ontology Consortium contribution), RNA 2008, doi: 10.1261/rna.657708

    C3'-endo:
        an individual ribose with δ between 55° and 110°
    C2'-endo:
        an individual ribose with δ between 120° and 175°
        
    Alternatively C3'- and C2'- endo can be defined using the pucker phase angle. C3'-endo [0°, 36°) as in canonical RNA and A-form DNA, and the C2'-endo [144°, 180°).

    Returns
    -------
    c3_endo : list
        '1' if if δ torsion angle forms a C3'-endo form, else '0'

    c2_endo : list
        '1' if if δ torsion angle forms a C2'-endo form, else '0'
    """

    c3_endo = []
    for _delta, _phase in zip(angle_d, angle_p):
        # C3 endo
        if (_delta >= 55 and _delta < 110) or (_phase >=0 and _phase < 36):
            c3_endo.append(1)
        else:
            c3_endo.append(0)

    c2_endo = []
    for _delta, _phase in zip(angle_d, angle_p):
        # C2 endo
        if (_delta >= 120 and _delta < 175) or (_phase >= 144 and _phase < 180):
            c2_endo.append(1)
        else:
            c2_endo.append(0)
        
    return c3_endo, c2_endo



def _intercalete(stacking_residue_index):        
    name = ""
    
    # 1-'0'-2-3
    # 2-'0'-1-3
    if [[0, 1], [0, 2]] == stacking_residue_index:
        name = "I0102"
            
    # 1-'2'-0-3
    # 0-'2'-1-3
    if [[0, 2], [1, 2]] == stacking_residue_index:
        name = "I0212"

    # 1-2-'0'-3
    if [[0, 2], [0, 3]] == stacking_residue_index:
        name = "I0203"

    # 0-'3'-1-2
    if [[0, 3], [1, 3]] == stacking_residue_index:
        name = "I0313"
            
    # 0-2-'1'-3
    # 0-3-'1'-2
    if [[1, 2], [1, 3]] == stacking_residue_index:
        name = "I1213"
        
    # 0-2-'3'-1
    # 0-1-'3'-2
    if [[1, 3], [2, 3]] == stacking_residue_index:
        name = "I1323"
        
    # 1-'2'-'0'-3
    if [[0, 2], [0, 3], [1, 2]] == stacking_residue_index:
        name = "I020312"        

    # 0-'2'-'1'-3
    if [[0, 2], [1, 2], [1, 3]] == stacking_residue_index:
        name = "I021213"

    # 0-'3'-'1'-2
    if [[0, 3], [1, 2], [1, 3]] == stacking_residue_index:
        name = "I031213"
    
    return name



def calc_annotation(init_pdb, traj, rnames):
    """
    Annotation
    https://github.com/srnas/barnaba/blob/master/examples/example_03_annotate.ipynb
    
    `stackings, pairings, res = bb.annotate(pdb)`  
     
    returns three lists:
    
    - a list of stacking interactions
    - a list of pairing interactions
    - the list of residue names following the usual convention RESNAME_RESNUMBER_CHAININDEX
    
    `stackings` and `pairings` contain the list of interactions for the N frames in the PDB/trajectory file and it is organized in the following way: for a given frame there are interactions between residues with index pairings`[i][0][k][0]` 
    and `pairings[i][0][k][1]`. The type of interaction is specified at the element `pairings[i][1][k]`.

    ### Decypher the annotation  ###
    Base-pairing are classified according to the Leontis-Westhof classification, where 
    - W = Watson-Crick edge
    - H = Hoogsteeen edge 
    - S= Sugar edge
    - c/t = cis/trans
    - XXx = when two bases are close in space, but they do not fall in any of the categories. This happens frequently for low-resolution structures or from molecular simulations.
    
    WWc pairs between complementary bases are called WCc or GUc.  
    Stacking are classified according to the MCannotate classification:
    - ">>" Upward
    - "<<" Downward 
    - "<>" Outward
    - "><" Inward
    
    
     
    ### Criteria for stacking/pairing ###
    First, we consider only bases that are "close" in space, i.e. $R_{ij} < 1.7$ and $R_{ji} < 1.7$.  
    $R_{ij} = (x_{ij}/5, y_{ij}/5, z_{ij}/3)$ is the SCALED position vector with components ${x,y,z}$ (in $\mathring{A}$) of base j constructed on base i.  
    The criteria for *base-stacking* are the following:
    
    $( |z_{ij}| \; AND \; |z_{ji}| > 2 \mathring{A} ) \; AND \;  
    (\rho_{ij} \; OR\; \rho_{ji} < 2.5 \mathring{A}) \; AND\;  
    (|\theta_{ij}| < 40^{\circ} ) $ 
    
    where
    - $ \rho_{ij} = \sqrt{x_{ij}^2 + y_{ij}^2} $  
    - $\theta_{ij}$ = angle between the vectors normal to the base plane
    
    The criteria for *base-pairing* are the following:  
    
    non stacked AND $|\theta_{ij}| < 60^{\circ}$ AND (number of hydrogen bonds $> 0$)  
    The number of hydrogen bonds is calculated as the number of donor-acceptor pairs with distance $< 3.3 \mathring{A}$. 
    If bases are complementary and the number of hydrogen bonds is > 1 (AU/GU) or > 2 (GC), the pair is considered WCc (or GUc).
    
    - cis/trans is calculated according to the value of the dihedral angle defined by $C1'_{i}-N1/N9_{i}-N1/N9_{j}-C1'_{j}$
    - edges are definded according to the value of $\psi = \arctan{(\hat{y}_{ij}/\hat{x}_{ij})}$. 
        1. Watson-Crick edge: $0.16 <\psi \le 2.0 rad$ 
        2. Hoogsteen edge:  $2.0 <\psi \le 4.0 rad $. 
        3. Sugar edge: $\psi > 4.0, \psi \le 0.16$
    
        
    **ATT!**
    - These criteria are slightly different from the one used in other popular software for annotating three-dimensional structures (e.g. X3DNA, MCAnnotate, Fr3D, etc.). From my experience, all these packages give slightly different results, 
    especially for non-Watson-Crick base-pairs.
    - Stacking is also problematic, as it relies on arbitrary criteria.
    - In all cases, criteria for stacking and pairing were calibrated to work well for high resolution structures. These criteria might not be optimal for low-resolution structures and to describe nearly-formed interactions such the ones that 
    are often encountered in molecular simulations.
    
    ### Dot-bracket annotation ###
    
    From the list of base-pairing, we can obtain the dot-bracket annotation using the function
    ```python
    dotbracket = bb.dot_bracket(pairings,res)
    ```
    this function returns a string for each frame in the PDB/simulation. Let's see this in action:
    

    ### Symbols for base stacking from François Major’s group ([RNA 3D Structure Course](https://docs.google.com/document/d/173tvcKJgAUmjd03zIKLz-_KCNTlcHBNDWlDfDtMhxmU/edit#) by Craig L. Zirbel and Neocles Leontis at Bowling Green State Universtiy)
    
    Two possible orientations of two stacked bases result in four base-stacking types: upward (>>), downward (<<), outward (<>) and inward (><). Two arrows pointing in the same direction (upward and downward) corresponds to the stacking type in 
    the canonical A-RNA double-helix. Upward or downward is chosen depending on which base is referred first (i.e. A>>B means B is stacked upward of A, or A is stacked downward of B). The two other types are less frequent in RNAs, respectively 
    inward (A><B; A or B is stacked inward of, respectively B or A) and outward (A<>B; A or B is stacked outward of, respectively B or A). 
    """        
    #
    # production
    #

    # stacking and pairing
    stackings, pairings, res = bb.annotate_traj(traj, stacking_rho_cutoff=4.0, stacking_angle_cutoff=45)

    # pucker
    pucker_angles, _ = bb.pucker_rao_traj(traj)
    pucker_angles = radian_to_degree(pucker_angles)  # [phase, amplitude]

    # backbone
    bb_angles, _ = bb.backbone_angles_traj(traj, )
    bb_angles = radian_to_degree(bb_angles) # [alpha, beta, gamma, delta, eps, zeta, chi]

    # define individual angles
    alpha = bb_angles[:,:,0]
    beta = bb_angles[:,:,1]
    gamma = bb_angles[:,:,2]
    delta = bb_angles[:,:,3]
    eps = bb_angles[:,:,4]
    zeta = bb_angles[:,:,5]
    chi = bb_angles[:,:,6]
    phase = pucker_angles[:,:,0]


    myclass = []
    for frame_idx in range(len(stackings)):
        """
        stackings[frame_idx] : list
            e.g. [[[0, 1], [1, 2], [2, 3]], ['>>', '>>', '>>']]
        """
        stacking_residue_index = stackings[frame_idx][0]
        stacking_pattern = stackings[frame_idx][1]
            
        # 1) Define c3'-endo and c2'-endo using δ torsion angles
        #    C3'-endo: an individual ribose with δ between 55° and 110°
        #    C2'-endo: an individual ribose with δ between 120° and 175°
        #    Ref: RNA backbone: Consensus all-angle conformers and modular string nomenclature (an RNA Ontology Consortium contribution), RNA 2008        
        c3_binary, c2_binary = [], []
        for _delta, _phase in zip(delta[frame_idx], phase[frame_idx]):
            if (_delta >= 55 and _delta < 110) or (_phase >=0 and _phase < 36):
                c3_binary.append(1)
            else:
                c3_binary.append(0)
            if (_delta >= 120 and _delta < 175) or (_phase >= 144 and _phase < 180):
                c2_binary.append(1)
            else:
                c2_binary.append(0)


        names = []


        #
        # A-form
        #
        if stacking_residue_index == [[0, 1], [1, 2], [2, 3]]:
            c3_binary, c2_binary = _check_endo(delta[frame_idx], phase[frame_idx])
            if stacking_pattern == ['>>', '>>', '>>'] and sum(c3_binary) == 4:
                names.append("AMa")
            else:
                names.append("AMi")

        #
        # Partial stacking
        #
        if stacking_residue_index == [[1, 2], [2, 3]] and stacking_pattern == ['>>', '>>']:
            names.append("F1")
        if stacking_residue_index == [[0, 1], [1, 2]] and stacking_pattern == ['>>', '>>']:
            names.append("F4")


        #
        # I: Intercaleted (nucleotide j inserts between and stacks against nb i and i+1)            
        #
        if len(stacking_pattern) >= 2:
            _name = _intercalete(stacking_residue_index)
            if _name.startswith("I"):
                names.append("I")


        #
        # Others
        #
        if len(stacking_pattern) == 0 or len(stacking_pattern) == 1:
            names.append("O")
        if len(names) == 0:
            names.append("O")

        assert len(names) == 1, "{}: multi-assigned: {}\t{}\t{}".format(frame_idx+1, names, stacking_pattern, stacking_residue_index)


        myclass.append(names[0])
        with open("log.txt", "a") as wf:
            wf.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(frame_idx+1, 
                                                                                                names[0], 
                                                                                                stacking_residue_index, 
                                                                                                stacking_pattern, 
                                                                                                list(alpha[frame_idx,:]), 
                                                                                                list(beta[frame_idx,:]), 
                                                                                                list(gamma[frame_idx,:]), 
                                                                                                list(delta[frame_idx,:]), 
                                                                                                list(eps[frame_idx,:]), 
                                                                                                list(zeta[frame_idx,:]), 
                                                                                                list(phase[frame_idx,:]), 
                                                                                                c3_binary, 
                                                                                                c2_binary))

    from collections import Counter
    d = Counter(myclass)
    mydata = {
        "AMa":  d["AMa"], \
        "AMi":  d["AMi"], \
        "I":   d["I"], \
        "F1":   d["F1"], \
        "F4":   d["F4"], \
        "O":    d["O"]
    }
    mydata = {
        "AMa": 100*d["AMa"]/len(myclass), \
        "AMi": 100*d["AMi"]/len(myclass), \
        "I":   100*d["I"]/len(myclass), \
        "F1":  100*d["F1"]/len(myclass), \
        "F4":  100*d["F4"]/len(myclass), \
        "O":   100*d["O"]/len(myclass)
    }

    mycolor = ["green", "blue", "red", "magenta", "orange", "black"]

    # rmsd scatter plot
    color = []
    for _ in myclass:
        if _ == "AMa":
            color.append("green")
        elif _ == "AMi":
            color.append("blue")
        elif _ == "I":
            color.append("red")
        elif _ in "F1":
            color.append("magenta")
        elif _ == "F4":
            color.append("orange")
        elif _ == "O":
            color.append("black")
        else:
            print("undefined {}".format(_))
            color.append("white")

    # recalculate rmsd
    init_traj = mdtraj.load(init_pdb)
    rmsd = list(bb.functions.rmsd_traj(init_traj, traj))   
    rmsd = np.array(rmsd) * UNIT_NM_TO_ANGSTROMS
    x = np.arange(1, len(rmsd)+1) * LOGGING_FREQUENCY * STRIDE * UNIT_PS_TO_NS * UNIT_PS_TO_NS # microsecond

    # define
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(32, 8), gridspec_kw={'width_ratios': [2, 1]})
    fig.suptitle(PLOT_TITLE, y=0.85)

    # xy-axis (1)
    ax1.set_xlabel(r'Time [${\rm \mu}$s]')
    ax1.xaxis.set_minor_locator(AutoMinorLocator())
    #ax1.set_xlim([0, len(x)]) 
    ax1.set_xlim([0, x.max()]) 
    ax1.set_ylabel(r'RMSD [${\rm \AA}$]')
    ax1.yaxis.set_minor_locator(AutoMinorLocator())
    ax1.set_ylim([0, 6])

    # xy-axis (2)
    ax2.set_ylabel('(%)')
    ax2.yaxis.set_label_position("right")
    ax2.yaxis.set_minor_locator(MultipleLocator(10))
    ax2.yaxis.set_ticks_position("right")
    ax2.set_ylim([0, 100])

    i = 0
    for k, v in mydata.items():
        ax2.text(x=i-0.3, y=v+3, s=f'{v:.1f}', size=24)
        i += 1

    # plot
    ax1.scatter(x, rmsd, color=color)
    ax2.bar(mydata.keys(), mydata.values(), width=1.0, color=mycolor)
    #plt.subplots_adjust(left=0.1,
    #                    bottom=0.1, 
    #                    right=0.9, 
    #                    top=0.9, 
    #                    wspace=0.4, 
    #                    hspace=0.4)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.1, wspace=0.1)
    plt.savefig("rmsd_conformation_population.png")



def calc_jcoupling(init_pdb, traj, rnames):
    """
    J-coupling
    """
    d_exp = {"beta1":   [0, 3.8, 3.9, 3.8],
            "beta2":   [0, 1.2, 0.5, 1.1],
            "gamma1":  [0, 1, 2.1, 2.0],
            "gamma2":  [0, 1, 1, 1.4],
            "epsilon": [8.8, 9.3, 9.3, 0],
            "nu1":     [1, 1, 1, 2.6],
            "nu2":     [3.6, 4.3, 4.7, 4.9],
            "nu3":     [8.7, 0, 8.8, 7.2]
            }

    d_rev = {"beta1":   [0, 3.6, 3.9, 3.5],
            "beta2":   [0, 2.1, 1.9, 2.3],
            "gamma1":  [2.6, 2.5, 2.6, 2.5],
            "gamma2":  [2.2, 1, 1.0, 1.1],
            "epsilon": [8.4, 8.8, 8.1, 0],
            "nu1":     [1.7, 1.6, 1.4, 3.0],
            "nu2":     [4.5, 4.5, 4.6, 5.0],
            "nu3":     [8.1, 8.1, 8.2, 6.4]
            }

    d_a14 = {"beta1":   [0, 3.5, 4.3, 2.5],
            "beta2":   [0, 2.1, 1.8, 3.2],
            "gamma1":  [2.6, 2.4, 2.7, 2.7],
            "gamma2":  [1.7, 1.3, 0.9, 0.9],
            "epsilon": [6.4, 6.6, 5.5, 0],
            "nu1":     [1.5, 3.0, 1.7, 3.3],
            "nu2":     [4.5, 4.8, 4.6, 5.3],
            "nu3":     [8.1, 6.2, 7.8, 5.9]
            }

    df_exp = pd.DataFrame(data=d_exp)
    df_rev = pd.DataFrame(data=d_rev)
    df_a14 = pd.DataFrame(data=d_a14)

    mydict={}
    mydict["H1H2"] = "nu1"
    mydict["H2H3"] = "nu2"
    mydict["H3H4"] = "nu3"
    mydict["1H5P"] = "beta1"
    mydict["2H5P"] = "beta2"
    mydict["1H5H4"] = "gamma1"
    mydict["2H5H4"] = "gamma2"
    mydict["H3P"] = "epsilon"

    couplings, rr = bb.jcouplings_traj(traj, couplings=["H1H2", "H2H3", "H3H4", "1H5P", "2H5P", "1H5H4", "2H5H4", "H3P"] ) # couplings: [frame idx, residue, jcoupling values]
    couplings[np.isnan(couplings)] = 0

    m = couplings.mean(axis=0)
    s = couplings.std(axis=0)

    pd.options.display.float_format = '{:.2f}'.format
    df_m = pd.DataFrame(m, columns=list(mydict.values()))
    df_s = pd.DataFrame(s, columns=list(mydict.values()))


    # plot
    import matplotlib.colors as mcolors
    mycolors = mcolors.TABLEAU_COLORS

    _, res = bb.backbone_angles(init_pdb)
    names = ["beta1", "beta2", "gamma1", "gamma2", "epsilon", "nu1", "nu2", "nu3"]
    fig, axes = plt.subplots(4,1, figsize=(18, 12), sharex=True)
    for i, r in enumerate(res):
        xpos = 0
        
        for name, mycolor in zip(names, mycolors):
            # hide values with zero
            exp_scale, rev_scale, a14_scale, scale = 1, 1, 1, 1
            if df_exp[name][i] == 0:
                exp_scale = 0
            if df_rev[name][i] == 0:
                rev_scale = 0
            if df_a14[name][i] == 0:
                a14_scale = 0
            if df_m[name][i] == 0:
                scale = 0
            
            axes[i].scatter(xpos-0.1, df_exp[name][i], marker='x', s=60 * exp_scale, c=mycolors[mycolor])
            axes[i].scatter(xpos, df_rev[name][i], marker='^', s=60 * rev_scale, c=mycolors[mycolor])
            axes[i].scatter(xpos+0.1, df_a14[name][i], marker='_', s=60 * a14_scale, c=mycolors[mycolor])
            axes[i].errorbar(xpos+0.2, df_m[name][i], yerr=df_s[name][i], fmt='o', capsize=6 * scale, markersize=10 * scale, c=mycolors[mycolor])

            # axes
            axes[i].set_title('C{}'.format(i+1), x=0.03, y=0.75, fontsize=24)
            axes[i].yaxis.set_minor_locator(AutoMinorLocator())
            axes[i].yaxis.set_ticks_position("left")
            axes[i].set_ylim(-1.5,13.5)
            axes[i].set_xlim(-0.5,7.5)
            
            # increment position
            xpos += 1

    axes[i].set_xticks([0,1,2,3,4,5,6,7], [r"$\beta$1",r"$\beta$2",r"$\gamma$1",r"$\gamma$2",r"$\epsilon$",r"$\nu$1",r"$\nu$2",r"$\nu$3"])        
    axes[i].set_ylabel("$^3$J (Hz)")
    axes[i].yaxis.set_label_coords(-0.1, 2)

    # create scatter and hide marker. this is for legend purpose.
    axes[i].scatter(xpos, xpos, marker='x', s=60, c="k", label="Experimental")
    axes[i].scatter(xpos, xpos, marker='^', s=60, c="k", label="DEShaw.Revised")
    axes[i].scatter(xpos, xpos, marker='_', s=60, c="k", label="DEShaw.Amber.OL3")
    axes[i].scatter(xpos, xpos, marker='o', s=60, c="k", label="Amber.OL3")
    axes[i].legend(bbox_to_anchor=(0.23, 4.05), fontsize=16)

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.1, wspace=0.1)
    plt.savefig("jcoupling.png")

    # save
    df_m.to_csv('coupling_summary_avg.txt', sep='\t', float_format='%.2f')
    df_s.to_csv('coupling_summary_std.txt', sep='\t', float_format='%.2f')



if __name__ == '__main__':

    PLOT_TITLE = "CCCC Amber ff14"
    UNIT_NM_TO_ANGSTROMS = 10
    UNIT_PS_TO_NS = 1/1000
    LOGGING_FREQUENCY = 100   # Default: 100 (unit: ps)
    STRIDE = 10               # Only read every stride-th frame. Each frame is saved 100 ps (LOGGING_FREQUENCY) as default.

    # initial structure
    init_pdb = "../../eq/solvated.pdb"

    # trajectory: 100ns each / checkpoint interval: 100ps
    #n = len(glob.glob("../md*"))
    n = 300
    #n = 1
    ncfiles = [ "../md" + str(i) + "/traj.nc" for i in range(1,n+1) ]
    traj = mdtraj.load(ncfiles, top=init_pdb, stride=STRIDE)

    rnames = [ residue.name for residue in traj.topology.residues if residue.name not in ["HOH", "NA", "CL"] ]

    # calculate and plot
    calc_bb_torsion(init_pdb, traj, rnames)
    calc_sugar_pucker(init_pdb, traj, rnames)
    #calc_rg(init_pdb, traj, rnames)
    #calc_rmsd(init_pdb, traj, rnames)
    calc_annotation(init_pdb, traj, rnames)
    calc_jcoupling(init_pdb, traj, rnames)





