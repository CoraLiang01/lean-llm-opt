import gurobipy as gp
import pandas as pd
import numpy as np
import re
sensor_sites_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant20/inputs/sensor_sites.csv'
zones_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant20/inputs/monitoring_zones.csv'
sensor_sites_df = pd.read_csv(sensor_sites_path, sep=',')
if sensor_sites_df['Center'].isnull().any():
    raise ValueError('Missing Center identifier in sensor_sites.csv')
if sensor_sites_df['OpeningCost'].isnull().any():
    raise ValueError('Missing OpeningCost in sensor_sites.csv')
if sensor_sites_df['CoveredDistricts'].isnull().any():
    raise ValueError('Missing CoveredDistricts in sensor_sites.csv')
zones_df = pd.read_csv(zones_path, sep=',')
if zones_df['Zone'].isnull().any():
    raise ValueError('Missing Zone identifier in monitoring_zones.csv')

def norm_id(x):
    return str(x).strip().casefold()
sensor_sites = list(sensor_sites_df['Center'])
zones = list(zones_df['Zone'])
site_covers = {}
for (idx, row) in sensor_sites_df.iterrows():
    site = row['Center']
    covered_str = row['CoveredDistricts']
    covered_zones = [z.strip() for z in str(covered_str).split(';') if z.strip()]
    site_covers[site] = set(covered_zones)
zone_set = set(zones)
for (site, covered) in site_covers.items():
    for z in covered:
        if z not in zone_set:
            raise ValueError(f"Sensor site {site} covers unknown zone '{z}' not in monitoring_zones.csv")
zone_covered_by = {z: set() for z in zones}
for (site, covered) in site_covers.items():
    for z in covered:
        zone_covered_by[z].add(site)
for z in zones:
    if not zone_covered_by[z]:
        raise ValueError(f"Zone '{z}' is not covered by any sensor site.")
cost = {}
for (idx, row) in sensor_sites_df.iterrows():
    site = row['Center']
    c = row['OpeningCost']
    if not np.issubdtype(type(c), np.integer) and (not np.issubdtype(type(c), np.floating)):
        raise ValueError(f'OpeningCost for site {site} is not numeric.')
    cost[site] = float(c)
m = gp.Model('SensorSetCover')
m.setParam('MIPGap', 0.0001)
y = m.addVars(sensor_sites, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((cost[site] * y[site] for site in sensor_sites)), gp.GRB.MINIMIZE)
for z in zones:
    m.addConstr(gp.quicksum((y[site] for site in zone_covered_by[z])) >= 1, name=f'cov_{z}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for site in sensor_sites:
        print(f'{y[site].VarName} {y[site].X}')
else:
    print(f'Solver status: {m.status}')