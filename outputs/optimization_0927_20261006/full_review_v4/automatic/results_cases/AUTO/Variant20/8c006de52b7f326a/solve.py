import gurobipy as gp
import pandas as pd
import numpy as np
import re
sensor_sites_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant20/inputs/sensor_sites.csv'
monitoring_zones_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant20/inputs/monitoring_zones.csv'
sensor_sites_df = pd.read_csv(sensor_sites_path, dtype=str, keep_default_na=False)
monitoring_zones_df = pd.read_csv(monitoring_zones_path, dtype=str, keep_default_na=False)
sensor_sites = sensor_sites_df['Center'].tolist()
try:
    opening_cost = {row['Center']: int(row['OpeningCost']) for (_, row) in sensor_sites_df.iterrows()}
except Exception as e:
    raise ValueError(f'Failed to convert OpeningCost to int for all sensor sites: {e}')
monitoring_zones = monitoring_zones_df['Zone'].tolist()
site_covers = {}
zone_covered_by = {zone: set() for zone in monitoring_zones}
for (_, row) in sensor_sites_df.iterrows():
    center = row['Center']
    covered_raw = row['CoveredDistricts']
    if covered_raw.strip() == '':
        covered_zones = []
    else:
        covered_zones = [z.strip() for z in covered_raw.split(';')]
    site_covers[center] = set(covered_zones)
    for zone in covered_zones:
        if zone in zone_covered_by:
            zone_covered_by[zone].add(center)
        else:
            pass
uncovered_zones = [zone for (zone, sites) in zone_covered_by.items() if len(sites) == 0]
if uncovered_zones:
    raise ValueError(f'The following monitoring zones are not covered by any sensor site: {uncovered_zones}')
m = gp.Model('MinimumCostSensorCover')
y_vars = m.addVars(sensor_sites, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((opening_cost[i] * y_vars[i] for i in sensor_sites)), gp.GRB.MINIMIZE)
for zone in monitoring_zones:
    covering_sites = zone_covered_by[zone]
    m.addConstr(gp.quicksum((y_vars[i] for i in covering_sites)) >= 1, name=f'cover_{zone}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f}')
    print('--- Selected Sensor Sites ---')
    for i in sensor_sites:
        if y_vars[i].X > 0.5:
            print(f'  Site {i}: INSTALLED (Cost: {opening_cost[i]})')
else:
    print(f'No optimal solution found. Status: {m.status}')