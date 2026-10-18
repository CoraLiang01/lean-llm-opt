import gurobipy as gp
import pandas as pd
import numpy as np
import re
sensor_sites_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant20/inputs/sensor_sites.csv'
monitoring_zones_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant20/inputs/monitoring_zones.csv'
sensor_sites_df = pd.read_csv(sensor_sites_path, dtype=str, keep_default_na=False)
for col in ['Center', 'OpeningCost', 'CoveredDistricts']:
    if col not in sensor_sites_df.columns:
        raise KeyError(f"Column '{col}' not found in sensor_sites.csv")
monitoring_zones_df = pd.read_csv(monitoring_zones_path, dtype=str, keep_default_na=False)
if 'Zone' not in monitoring_zones_df.columns:
    raise KeyError("Column 'Zone' not found in monitoring_zones.csv")
sensor_sites = sensor_sites_df['Center'].tolist()
monitoring_zones = monitoring_zones_df['Zone'].tolist()
opening_cost = {}
for idx, row in sensor_sites_df.iterrows():
    site = row['Center']
    try:
        cost = int(row['OpeningCost'])
    except Exception as e:
        raise ValueError(f"Invalid OpeningCost for site '{site}': {row['OpeningCost']}")
    opening_cost[site] = cost
site_covers = {}
for idx, row in sensor_sites_df.iterrows():
    site = row['Center']
    covered_raw = row['CoveredDistricts']
    covered_zones = [z.strip() for z in covered_raw.split(';') if z.strip() != '']
    site_covers[site] = set(covered_zones)
zone_covered_by = {zone: set() for zone in monitoring_zones}
for site in sensor_sites:
    for zone in site_covers[site]:
        if zone in zone_covered_by:
            zone_covered_by[zone].add(site)
for zone in monitoring_zones:
    if len(zone_covered_by[zone]) == 0:
        raise ValueError(f"Monitoring zone '{zone}' is not covered by any sensor site.")
m = gp.Model('MinimumCostSensorCover')
y_vars = m.addVars(sensor_sites, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((opening_cost[site] * y_vars[site] for site in sensor_sites)), gp.GRB.MINIMIZE)
for zone in monitoring_zones:
    covered_sites = zone_covered_by[zone]
    m.addConstr(gp.quicksum((y_vars[site] for site in covered_sites)) >= 1, name=f'cover_{zone}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f}')
    print('--- Selected Sensor Sites ---')
    for site in sensor_sites:
        if y_vars[site].X > 0.5:
            print(f'  Site {site}: SELECTED (cost {opening_cost[site]})')
else:
    print(f'No optimal solution found. Status: {m.status}')