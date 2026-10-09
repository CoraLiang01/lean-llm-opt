import gurobipy as gp
import pandas as pd
import numpy as np
import re
sensor_sites_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant20/inputs/sensor_sites.csv'
zones_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant20/inputs/monitoring_zones.csv'
sensor_sites_df = pd.read_csv(sensor_sites_path, sep=',')
zones_df = pd.read_csv(zones_path, sep=',')

def norm_id(x):
    return str(x).strip()
sites = [norm_id(c) for c in sensor_sites_df['Center']]
zones = [norm_id(z) for z in zones_df['Zone']]
site_covers = {}
for (idx, row) in sensor_sites_df.iterrows():
    site = norm_id(row['Center'])
    covered = [norm_id(z) for z in str(row['CoveredDistricts']).split(';')]
    site_covers[site] = set(covered)
zone_covered_by = {z: set() for z in zones}
for site in sites:
    for z in site_covers[site]:
        if z in zone_covered_by:
            zone_covered_by[z].add(site)
uncovered_zones = [z for z in zones if len(zone_covered_by[z]) == 0]
if uncovered_zones:
    raise ValueError(f'The following zones are not covered by any site: {uncovered_zones}')
site_cost = {}
for (idx, row) in sensor_sites_df.iterrows():
    site = norm_id(row['Center'])
    if site in sites:
        site_cost[site] = float(row['OpeningCost'])
if set(site_cost.keys()) != set(sites):
    raise ValueError('Mismatch between site cost keys and site list.')
m = gp.Model('SensorSetCover')
y = m.addVars(sites, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((site_cost[site] * y[site] for site in sites)), gp.GRB.MINIMIZE)
for z in zones:
    covered_sites = zone_covered_by[z]
    m.addConstr(gp.quicksum((y[site] for site in covered_sites)) >= 1, name=f'cover_{z}')
m.Params.MIPGap = 0.0001
m.optimize()