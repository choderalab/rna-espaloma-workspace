#!/usr/bin/env python
# coding: utf-8
import os, sys, shutil
import pathlib
import glob as glob
import numpy as np
import re
import warnings
import mdtraj as md
import click
import openmmtools as mmtools
from openmm.app import *
from openmm import *
from openmm.unit import *
from openff.toolkit.utils import utils as offutils
from sys import stdout
from openmm.app import PDBFile
from pdbfixer import PDBFixer
from mdtraj.reporters import NetCDFReporter

# Export version
import openmmforcefields
import openff.toolkit
print(f"openmmforcefield: {openmmforcefields.__version__}")
print(f"openff-toolkit: {openff.toolkit.__version__}")


def create_position_restraint(position, restraint_atom_indices):
    """
    heavy atom restraint
    """
    force = CustomExternalForce("k*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
    force.addGlobalParameter("k", 10.0*kilocalories_per_mole/angstroms**2)
    force.addPerParticleParameter("x0")
    force.addPerParticleParameter("y0")
    force.addPerParticleParameter("z0")
    for i in restraint_atom_indices:
        atom_crd = position[i]
        force.addParticle(i, atom_crd.value_in_unit(nanometers))
    return force



def export_xml(simulation, system):
    """
    Save state as XML
    """
    state = simulation.context.getState(getPositions=True, getVelocities=True, getEnergy=True, getForces=True)
    # Save and serialize the final state
    with open("state.xml", "w") as wf:
        xml = XmlSerializer.serialize(state)
        wf.write(xml)

    # Save and serialize integrator
    with open("integrator.xml", "w") as wf:
        xml = XmlSerializer.serialize(simulation.integrator)
        wf.write(xml)

    # Save the final state as a PDB
    with open("state.pdb", "w") as wf:
        PDBFile.writeFile(
            simulation.topology,
            simulation.context.getState(
                getPositions=True,
                enforcePeriodicBox=True).getPositions(),
                file=wf,
                keepIds=True
        )

    # Save and serialize system
    system.setDefaultPeriodicBoxVectors(*state.getPeriodicBoxVectors())
    with open("system.xml", "w") as wf:
        xml = XmlSerializer.serialize(system)
        wf.write(xml)


def run(**options):
    #print(options)
    pdbfile = options["pdbfile"]
    output_prefix = options["output_prefix"]
    water_model = options["water_model"]
    box_padding = 12.0 * angstrom
    salt_conc = 0.08 * molar
    nb_cutoff = 10 * angstrom
    hmass = 3.5 * amu
    timestep = 4 * femtoseconds    
    temperature = 300 * kelvin
    pressure = 1 * atmosphere
    nsteps_min = 100
    nsteps_eq = 125000   # 500ps
    nsteps_prod = 2500000  # 10ns
    checkpoint_frequency = 250000  # 1ns
    logging_frequency = 25000  # 100ps
    netcdf_frequency = 25000  # 100ps


    platform = mmtools.utils.get_fastest_platform()
    platform_name = platform.getName()
    print("fastest platform is ", platform_name)
    if platform_name == "CUDA":
        # Set CUDA DeterministicForces (necessary for MBAR)
        platform.setPropertyDefaultValue('DeterministicForces', 'true')
        platform.setPropertyDefaultValue('Precision', 'mixed')
    else:
        #raise Exception("fastest platform is not CUDA")
        warnings.warn("fastest platform is not CUDA")

    

    """
    create system
    """
    # pdbfixer: fix structure if necessary
    fixer = PDBFixer(filename=pdbfile)
    fixer.findMissingResidues()
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.removeHeterogens(True)
    fixer.findMissingAtoms()
    fixer.addMissingHydrogens(7.0)  # default: 7
    PDBFile.writeFile(fixer.topology, fixer.positions, open('pdbfixer.pdb', 'w'))

    # 3 point water model
    if water_model == 'tip3p':
        water_model='tip3p'
        ff = ForceField('amber/RNA.OL3.xml', 'amber/protein.ff14SB.xml', 'amber/tip3p_standard.xml', 'amber/tip3p_HFE_multivalent.xml')
    elif water_model == 'tip3pfb':
        water_model='tip3p'
        ff = ForceField('amber/RNA.OL3.xml', 'amber/protein.ff14SB.xml', 'amber/tip3pfb_standard.xml', 'amber/tip3pfb_HFE_multivalent.xml')      
    elif water_model == 'spce':
        water_model='tip3p'
        ff = ForceField('amber/RNA.OL3.xml', 'amber/protein.ff14SB.xml', 'amber/spce_standard.xml', 'amber/spce_HFE_multivalent.xml')  
    # https://github.com/openmm/openmmforcefields/issues/272
    #elif water_model == 'opc3':
    #    water_model='tip3p'
    #    ff = ForceField('amber/RNA.OL3.xml', 'amber/protein.ff14SB.xml', 'amber/opc3_standard.xml')
    # 4 point water model
    elif water_model == 'tip4pew':
        water_model='tip4pew'
        ff = ForceField('amber/RNA.OL3.xml', 'amber/protein.ff14SB.xml', 'amber/tip4pew_standard.xml', 'amber/tip4pew_HFE_multivalent.xml')
    elif water_model == 'tip4pfb':
        water_model='tip4pew'
        ff = ForceField('amber/RNA.OL3.xml', 'amber/protein.ff14SB.xml', 'amber/tip4pfb_standard.xml', 'amber/tip4pfb_HFE_multivalent.xml')
    elif water_model == 'opc':
        water_model='tip4pew'
        # Use a custom xml file. Original xml file contains Uranium (U) which conflicts with Uridine (U) residue.
        ff = ForceField('amber/RNA.OL3.xml', 'amber/protein.ff14SB.xml', '/home/takabak/.ffxml/amber/opc_standard.xml')

    # solvate system
    modeller = Modeller(fixer.topology, fixer.positions)
    modeller.addSolvent(ff, model=water_model, padding=box_padding, ionicStrength=salt_conc)
    PDBFile.writeFile(modeller.topology, modeller.positions, file=open('solvated.pdb', 'w'))

    # create system
    system = ff.createSystem(modeller.topology, nonbondedMethod=PME, nonbondedCutoff=nb_cutoff, constraints=HBonds, rigidWater=True, hydrogenMass=hmass)
    integrator = LangevinMiddleIntegrator(temperature, 1/picosecond, timestep)
    simulation = Simulation(modeller.topology, system, integrator)
    simulation.context.setPositions(modeller.positions)

    # define reporter
    simulation.reporters.append(NetCDFReporter(os.path.join(output_prefix, 'traj.nc'), netcdf_frequency))
    simulation.reporters.append(CheckpointReporter(os.path.join(output_prefix, 'checkpoint.chk'), checkpoint_frequency))
    simulation.reporters.append(StateDataReporter(os.path.join(output_prefix, 'reporter.log'), logging_frequency, step=True, potentialEnergy=True, kineticEnergy=True, totalEnergy=True, temperature=True, volume=True, density=True, speed=True))

    # minimization
    restraint_atom_indices = [ a.index for a in modeller.topology.atoms() if a.residue.name in ['A', 'C', 'U', 'T'] and a.element.symbol != 'H' ]
    restraint_index = system.addForce(create_position_restraint(modeller.positions, restraint_atom_indices))

    simulation.minimizeEnergy(maxIterations=nsteps_min)
    minpositions = simulation.context.getState(getPositions=True).getPositions()    
    PDBFile.writeFile(modeller.topology, minpositions, open(os.path.join(output_prefix, 'min.pdb'), 'w'))   


    # Equilibration
    # Heating
    n = 50
    for i in range(n):
        temp = temperature * i / n
        simulation.context.setVelocitiesToTemperature(temp)    # initialize velocity
        integrator.setTemperature(temp)    # set target temperature
        simulation.step(int(nsteps_eq/n))

    # NVT
    integrator.setTemperature(temperature)
    simulation.step(nsteps_eq)

    # NPT
    system.removeForce(restraint_index)
    system.addForce(MonteCarloBarostat(pressure, temperature))
    simulation.context.reinitialize(preserveState=True)
    simulation.step(nsteps_prod)

    """
    Export state in xml format
    """
    export_xml(simulation, system)



@click.command()
@click.option('--pdbfile', required=True, help='path to input pdb file')
@click.option('--output_prefix', default=".", help='path to output files')
#@click.option('--water_model', '-w', type=click.Choice(['tip3p', 'tip3pfb', 'spce', 'opc3', 'tip4pew', 'tip4pfb', 'opc']), help='water model')
@click.option('--water_model', '-w', type=click.Choice(['tip3p', 'tip3pfb', 'spce', 'tip4pew', 'tip4pfb', 'opc']), help='water model')
def cli(**kwargs):
    run(**kwargs)



if __name__ == "__main__":
    cli()