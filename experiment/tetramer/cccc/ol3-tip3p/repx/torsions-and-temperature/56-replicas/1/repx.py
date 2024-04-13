#!/usr/bin/env python
# coding: utf-8
import os, sys, math
import numpy as np
import click
import inspect
from sys import stdout
import glob
import tempfile
from openmm import *
from openmm.app import *
from openmm.unit import *
from openmmtools import testsystems, states, mcmc, forces, alchemy
from openmmtools.utils import get_fastest_platform
from openmmtools.multistate import ReplicaExchangeSampler, MultiStateReporter
import mdtraj
import logging
import datetime
import warnings
import barnaba as bb
from barnaba import definitions
from barnaba.nucleic import Nucleic



def run(**options):
    #------------------------------------
    #system_options = {}
    #system_options['nonbondedMethod'] = PME
    #system_options['ewaldErrorTolerance'] = 0.0005    # default: 0.0005
    #system_options['nonbondedCutoff'] = 10 * angstroms  # default: 10 angstroms
    #system_options['rigidWater'] = True                # default: 
    #system_options['constraints'] = HBonds
    #system_options['hydrogenMass'] = 3.5 * amu

    default_pressure = 1 * atmosphere
    default_temperature = 278 * kelvin
    default_timestep = 4 * femtosecond
    default_collision_rate = 1/picosecond
    default_swap_scheme = 'swap-all'
    default_steps_per_replica = 250   # 1 ps/replica
    default_number_of_iterations = 10000  # 10 ns/replica
    default_checkpoint_interval = 1

    temperature_range_1 = 278
    temperature_range_2 = 396
    temperature_ratio = temperature_range_2/temperature_range_1
    n_temperature_lambda = 48
    
    torsion_range_1 = 1
    torsion_range_2 = 0.3
    n_torsion_lambda = 8
    
    protocol = {}
    protocol['temperature']    = np.concatenate( [np.linspace(1,1,n_torsion_lambda), np.linspace(1,temperature_ratio,n_temperature_lambda+1)[1:]] ) * default_temperature
    protocol['lambda_torsions'] = np.concatenate( [np.linspace(torsion_range_2, torsion_range_1, n_torsion_lambda)[::-1], np.linspace(torsion_range_2,torsion_range_2,n_temperature_lambda+1)[1:]] )
    #------------------------------------
    

    #-----
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
        "HO3'", \
    ]
    #-----


    input_prefix = options["input_prefix"]
    

    platform = get_fastest_platform()
    platform_name = platform.getName()
    if platform_name == "CUDA":
        # Set CUDA DeterministicForces (necessary for MBAR)
        platform.setPropertyDefaultValue('DeterministicForces', 'true')
        platform.setPropertyDefaultValue('Precision', 'mixed')
    else:
        #raise Exception("fastest platform is not CUDA")
        warnings.warn("fastest platform is not CUDA")
        #print("Fastest platform is {}".format(platorm_name), file=sys.stdout)
    

    # Deserialize system file and load system
    with open(os.path.join(input_prefix, 'system.xml'), 'r') as f:
        system = XmlSerializer.deserialize(f.read())

    # Deserialize integrator file and load integrator
    with open(os.path.join(input_prefix, 'integrator.xml'), 'r') as f:
        integrator = XmlSerializer.deserialize(f.read())

    # Set up simulation 
    pdb = PDBFile(os.path.join(input_prefix, 'state.pdb'))
    simulation = Simulation(pdb.topology, system, integrator, platform)

    # Load state
    with open(os.path.join(input_prefix, 'state.xml'), 'r') as f:
        state_xml = f.read()
    state = XmlSerializer.deserialize(state_xml)
    simulation.context.setState(state)

    # Get atom indices (multistatereporter)
    top = mdtraj.load_pdb(os.path.join(input_prefix, 'state.pdb'))
    res = [ r for r in top.topology.residues if r.name not in ("HOH", "NA", "CL") ]
    atom_indices, backbone_atom_indices = [], []
    for r in res:
        for a in r.atoms:
            atom_indices.append(a.index)
            # ------
            # store backbone atom indices to specify torsion indices relevant to backbones
            # ------
            #if a.name in backbone_sugar_atoms:
            #    backbone_atom_indices.append(a.index)

    # Get torsion indices (alchemical region)
    torsion_indices = []
    forces = list(system.getForces())
    for force in forces:
        name = force.__class__.__name__
        if "Torsion" in name:
            for i in range(force.getNumTorsions()):
                id1, id2, id3, id4, periodicity, phase, k = force.getTorsionParameters(i)
                torsion_indices.append(i)
                # -----
                # specify backbone torsions only
                # -----
                #if id1 in backbone_atom_indices and id2 in backbone_atom_indices and id3 in backbone_atom_indices and id4 in backbone_atom_indices:
                    #print(i, force.getTorsionParameters(i), top.topology.atom(id1), top.topology.atom(id2), top.topology.atom(id3), top.topology.atom(id4))
                    #torsion_indices.append(i)

    # Define alchemical region and thermodynamic state
    #alchemical_region = alchemy.AlchemicalRegion(alchemical_torsions=torsion_indices)
    alchemical_region = alchemy.AlchemicalRegion(alchemical_atoms=atom_indices, alchemical_torsions=torsion_indices)
    factory = alchemy.AbsoluteAlchemicalFactory()
    alchemical_system = factory.create_alchemical_system(system, alchemical_region)
    alchemical_state = alchemy.AlchemicalState.from_system(alchemical_system)
    thermodynamic_states = states.create_thermodynamic_state_protocol(alchemical_system, protocol=protocol, composable_states=[alchemical_state])
    sampler_states = states.SamplerState(positions=state.getPositions(), box_vectors=state.getPeriodicBoxVectors())

    # Rename old storge file
    storage_file = 'enhanced.nc'
    n = glob.glob(storage_file + '.nc')
    if os.path.exists(storage_file):
        #print('{} already exists. File will be renamed but checkpoint files will be deleted'.format(storage_file))
        os.remove('enhanced_checkpoint.nc')
        os.rename(storage_file, storage_file + "{}".format(str(len(n))))

    # ----
    # Replica exchange sampler
    # LangevinSplittingDynamicsMove: High-quality Langevin integrator family based on symmetric Strang splittings, using g-BAOAB as default
    # BAOAB integrator (i.e. with V R O R V splitting), which was shown empirically to add a very small integration error in configurational space
    # https://github.com/openmm/openmm/issues/2520
    # https://openmmtools.readthedocs.io/en/stable/gettingstarted.html
    # https://openmmtools.readthedocs.io/en/stable/gettingstarted.html
    # ----

    #move =  mcmc.LangevinSplittingDynamicsMove(timestep=default_timestep, \
    #                                           n_steps=default_steps_per_replica, \
    #                                           collision_rate=default_collision_rate, \
    #                                           reassign_velocities=False, \
    #                                           splitting='V R O R V')  # default: "V R O R V"

    mcmc_move = mcmc.LangevinDynamicsMove(timestep=default_timestep, collision_rate=default_collision_rate, reassign_velocities=False, n_steps=default_steps_per_replica, n_restart_attempts=6)
    simulation = ReplicaExchangeSampler(mcmc_moves=mcmc_move, number_of_iterations=default_number_of_iterations, replica_mixing_scheme=default_swap_scheme, online_analysis_interval=None)
    #reporter = MultiStateReporter(storage='enhanced.nc', checkpoint_interval=default_checkpoint_interval)
    reporter = MultiStateReporter(storage='enhanced.nc', checkpoint_interval=default_checkpoint_interval, analysis_particle_indices=atom_indices)
    simulation.create(thermodynamic_states, sampler_states=sampler_states, storage=reporter)

    # Just to check if the performance is better using this - for Openmm <= 7.7
    from openmmtools.cache import ContextCache
    simulation.energy_context_cache = ContextCache(capacity=None, time_to_live=None, platform=platform)
    simulation.sampler_context_cache = ContextCache(capacity=None, time_to_live=None, platform=platform)

    # Run!
    simulation.run()



@click.command()
@click.option('--input_prefix', default='../eq', help='path to load xml files to create systems and pdb file to read topology')
def cli(**kwargs):
    run(**kwargs)



if __name__ == "__main__":
    cli()
