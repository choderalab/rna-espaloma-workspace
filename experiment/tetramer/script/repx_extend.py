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
    ncfile = options["input_ncfile"]
    n_iterations = options["n_iterations"]

    platform = get_fastest_platform()
    platform_name = platform.getName()
    if platform_name == "CUDA":
        # Set CUDA DeterministicForces (necessary for MBAR)
        platform.setPropertyDefaultValue('DeterministicForces', 'true')
        platform.setPropertyDefaultValue('Precision', 'mixed')
    else:
        warnings.warn("fastest platform is not CUDA")

    import openmmtools as mmtools
    reporter = mmtools.multistate.MultiStateReporter(ncfile, open_mode='r')
    simulation = ReplicaExchangeSampler.from_storage(reporter)
    
    # Just to check if the performance is better using this - for Openmm <= 7.7
    from openmmtools.cache import ContextCache
    simulation.energy_context_cache = ContextCache(capacity=None, time_to_live=None, platform=platform)
    simulation.sampler_context_cache = ContextCache(capacity=None, time_to_live=None, platform=platform)

    # Run!
    simulation.extend(n_iterations=n_iterations)



@click.command()
@click.option('--input_ncfile', default='enhanced.nc', help='trajectory filename to resume simulation')
@click.option('--n_iterations', default=100, type=int, help='number of iterations')
def cli(**kwargs):
    run(**kwargs)



if __name__ == "__main__":
    cli()