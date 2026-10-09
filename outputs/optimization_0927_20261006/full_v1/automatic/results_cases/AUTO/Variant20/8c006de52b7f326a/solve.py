import gurobipy as gp
import pandas as pd
import numpy as np
import re
sensor_sites_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant20/inputs/sensor_sites.csv'
monitoring_zones_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant20/inputs/monitoring_zones.csv'
sensor_sites_df = pd.read_csv(sensor_sites_path, dtype=str, keep_default_na=False)
monitoring_zones_df = pd.read_csv(monitoring_zones_path, dtype=str, keep_default_na=False)
site_ids = sensor_sites_df['Center'].tolist()
zone_ids = monitoring_zones_df['Zone'].tolist()
opening_cost = {}
for (idx, row) in sensor_sites_df.iterrows():
    site = row['Center']
    try:
        opening_cost[site] = int(row['OpeningCost'])
    except Exception as e:
        raise ValueError(f"Invalid OpeningCost for site {site}: {row['OpeningCost']}") from e
site_covers = {}
for (idx, row) in sensor_sites_df.iterrows():
    site = row['Center']
    covered_str = row['CoveredDistricts']
    covered_zones = [z.strip() for z in covered_str.split(';') if z.strip() != '']
    site_covers[site] = set(covered_zones)
zone_covered_by = {zone: set() for zone in zone_ids}
for site in site_ids:
    for zone in site_covers[site]:
        if zone in zone_covered_by:
            zone_covered_by[zone].add(site)
for zone in zone_ids:
    if len(zone_covered_by[zone]) == 0:
        raise ValueError(f'Zone {zone} is not covered by any site.')
m = gp.Model('SensorSetCover')
y_vars = m.addVars(site_ids, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((opening_cost[site] * y_vars[site] for site in site_ids)), gp.GRB.MINIMIZE)
for zone in zone_ids:
    covering_sites = zone_covered_by[zone]
    m.addConstr(gp.quicksum((y_vars[site] for site in covering_sites)) >= 1, name=f'cover_{zone}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f}')
    print('--- Selected Sensor Sites ---')
    for site in site_ids:
        if y_vars[site].X > 0.5:
            print(f'  Site {site}: INSTALLED (cost {opening_cost[site]})')
else:
    print(f'No optimal solution found. Status: {m.status}')