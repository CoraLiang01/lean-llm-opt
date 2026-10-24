import gurobipy as gp
import pandas as pd
import numpy as np
import re
sensor_sites_path = 'benchmark_dataset/Variant20/inputs/sensor_sites.csv'
sensor_sites_df = pd.read_csv(sensor_sites_path, sep=',', dtype=str, keep_default_na=False)
monitoring_zones_path = 'benchmark_dataset/Variant20/inputs/monitoring_zones.csv'
monitoring_zones_df = pd.read_csv(monitoring_zones_path, sep=',', dtype=str, keep_default_na=False)
sensor_sites_df['Center'] = sensor_sites_df['Center'].str.strip()
site_ids = sensor_sites_df['Center'].tolist()
try:
    sensor_sites_df['OpeningCost'] = sensor_sites_df['OpeningCost'].astype(int)
except Exception as e:
    raise ValueError(f'Failed to convert OpeningCost to int: {e}')
opening_costs = dict(zip(sensor_sites_df['Center'], sensor_sites_df['OpeningCost']))
monitoring_zones_df['Zone'] = monitoring_zones_df['Zone'].str.strip()
zone_ids = monitoring_zones_df['Zone'].tolist()
site_covers = {}
zone_covered_by = {z: set() for z in zone_ids}
for (idx, row) in sensor_sites_df.iterrows():
    site = row['Center']
    covered_raw = row['CoveredDistricts']
    covered_zones = [z.strip() for z in covered_raw.split(';') if z.strip() != '']
    site_covers[site] = set(covered_zones)
    for z in covered_zones:
        if z in zone_covered_by:
            zone_covered_by[z].add(site)
        else:
            raise ValueError(f"Zone '{z}' covered by site '{site}' not found in monitoring_zones.csv")
for z in zone_ids:
    if len(zone_covered_by[z]) == 0:
        raise ValueError(f"Zone '{z}' is not covered by any sensor site.")
m = gp.Model('MinimumCostSensorCover')
y_vars = m.addVars(site_ids, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((opening_costs[i] * y_vars[i] for i in site_ids)), gp.GRB.MINIMIZE)
for z in zone_ids:
    m.addConstr(gp.quicksum((y_vars[i] for i in zone_covered_by[z])) >= 1, name=f'cover_{z}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f}')
    print('--- Selected Sensor Sites ---')
    for i in site_ids:
        if y_vars[i].X > 0.5:
            print(f'  Site {i}: SELECTED (Cost: {opening_costs[i]})')
else:
    print(f'No optimal solution found. Status: {m.status}')