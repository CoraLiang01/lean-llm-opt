import gurobipy as gp
import pandas as pd
import numpy as np
import re
sensor_sites_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant20/inputs/sensor_sites.csv'
sensor_df = pd.read_csv(sensor_sites_path, sep=',')
zones_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant20/inputs/monitoring_zones.csv'
zones_df = pd.read_csv(zones_path, sep=',')

def norm_id(x):
    return str(x).strip()
sites = [norm_id(c) for c in sensor_df['Center']]
zones = [norm_id(z) for z in zones_df['Zone']]
opening_cost = {}
for (idx, row) in sensor_df.iterrows():
    site = norm_id(row['Center'])
    cost = row['OpeningCost']
    opening_cost[site] = cost
site_covers = {}
for (idx, row) in sensor_df.iterrows():
    site = norm_id(row['Center'])
    covered_str = str(row['CoveredDistricts'])
    covered_zones = [norm_id(z) for z in covered_str.split(';') if z.strip()]
    site_covers[site] = set(covered_zones)
zone_covered_by = {z: set() for z in zones}
for site in sites:
    for z in site_covers[site]:
        if z in zone_covered_by:
            zone_covered_by[z].add(site)
uncovered_zones = [z for z in zones if len(zone_covered_by[z]) == 0]
if uncovered_zones:
    raise ValueError(f'The following zones are not covered by any site: {uncovered_zones}')
m = gp.Model('SensorSetCover')
y = m.addVars(sites, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((opening_cost[site] * y[site] for site in sites)), gp.GRB.MINIMIZE)
for z in zones:
    covering_sites = zone_covered_by[z]
    m.addConstr(gp.quicksum((y[site] for site in covering_sites)) >= 1, name=f'cover_{z}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f}')
    print('--- Selected Sensor Sites ---')
    for site in sites:
        if y[site].X > 0.5:
            print(f'  Site {site}: INSTALLED (cost {opening_cost[site]})')
else:
    print(f'No optimal solution found. Status: {m.status}')