#!/usr/bin/env python
# coding: utf-8

# YANK Simulation Health Report
# General Settings
# ========
# 
# Mandatory Settings
# ----------------
# * `store_directory`: Location where the experiment was run. This has an `analysis.yaml` file and two `.nc` files.
# 
# Optional Settings
# ----------------
# * `decorrelation_threshold`: When number of decorrelated samples is less than this percent of the total number of samples, raise a warning. Default: `0.1`.
# * `mixing_cutoff`: Minimal level of mixing percent from state `i` to `j` that will be plotted. Default: `0.05`.
# * `mixing_warning_threshold`: Level of mixing where transition from state `i` to `j` generates a warning based on percent of total swaps. Default: `0.90`.
# * `phase_stacked_replica_plots`: Boolean to set if the two phases' replica mixing plots should be stacked one on top of the other or side by side. If `True`, every replica will span the whole notebook, but the notebook will be longer. If `False`, the two phases' plots will be next to each other for a shorter notebook, but a more compressed view. Default `False`.


# Mandatory Settings
store_directory = './'
analyzer_kwargs = {}

# Optional Settings
decorrelation_threshold = 0.1
mixing_cutoff = 0.05
mixing_warning_threshold = 0.90
phase_stacked_replica_plots = False


# https://github.com/yaml/pyyaml/wiki/PyYAML-yaml.load(input)-Deprecation
import yaml
yaml.warnings({'YAMLLoadWarning': False})



from matplotlib import pyplot as plt
from yank.reports import notebook
#get_ipython().run_line_magic('matplotlib', 'inline')
report = notebook.HealthReportData(store_directory, **analyzer_kwargs)
report.report_version()

print(report.general_simulation_data())
print(report.get_equilibration_data())
