import gurobipy as gp
import pandas as pd
import numpy as np
import re
sensor_sites_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant20/inputs/sensor_sites.csv'
monitoring_zones_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant20/inputs/monitoring_zones.csv'
sensor_sites_df = pd.read_csv(sensor_sites_path, dtype=str, keep_default_na=False)
monitoring_zones_df = pd.read_csv(monitoring_zones_path, dtype=str, keep_default_na=False)
sensor_sites = sensor_sites_df['Center'].tolist()
monitoring_zones = monitoring_zones_df['Zone'].tolist()
sensor_sites_df['OpeningCost'] = sensor_sites_df['OpeningCost'].astype(int)
opening_cost = dict(zip(sensor_sites_df['Center'], sensor_sites_df['OpeningCost']))

def parse_covered_zones(s):
    return [z.strip() for z in s.split(';') if z.strip()]
site_covers = {}
for (idx, row) in sensor_sites_df.iterrows():
    site = row['Center']
    covered = parse_covered_zones(row['CoveredDistricts'])
    site_covers[site] = set(covered)
zone_covered_by = {zone: set() for zone in monitoring_zones}
for site in sensor_sites:
    for zone in site_covers[site]:
        if zone in zone_covered_by:
            zone_covered_by[zone].add(site)
for zone in monitoring_zones:
    if not zone_covered_by[zone]:
        raise ValueError(f"Monitoring zone '{zone}' is not covered by any sensor site.")
m = gp.Model('MinimumCostSensorCover')
y_vars = m.addVars(sensor_sites, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((opening_cost[site] * y_vars[site] for site in sensor_sites)), gp.GRB.MINIMIZE)
for zone in monitoring_zones:
    m.addConstr(gp.quicksum((y_vars[site] for site in zone_covered_by[zone])) >= 1, name=f'cover_{zone}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total installation cost: {m.objVal:.2f}')
    print('--- Selected Sensor Sites ---')
    for site in sensor_sites:
        if y_vars[site].X > 0.5:
            print(f'  Site {site}: INSTALLED (cost {opening_cost[site]})')
else:
    print(f'No optimal solution found. Status: {m.status}')