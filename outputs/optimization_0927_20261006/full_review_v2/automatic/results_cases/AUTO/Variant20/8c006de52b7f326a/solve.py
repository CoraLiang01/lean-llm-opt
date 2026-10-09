import gurobipy as gp
import pandas as pd
import numpy as np
import re
sensor_sites_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant20/inputs/sensor_sites.csv'
monitoring_zones_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant20/inputs/monitoring_zones.csv'
sensor_sites_df = pd.read_csv(sensor_sites_path, dtype=str, keep_default_na=False)
monitoring_zones_df = pd.read_csv(monitoring_zones_path, dtype=str, keep_default_na=False)
sensor_sites_df['Center_norm'] = sensor_sites_df['Center'].str.strip()
site_ids = sensor_sites_df['Center_norm'].tolist()
monitoring_zones_df['Zone_norm'] = monitoring_zones_df['Zone'].str.strip()
zone_ids = monitoring_zones_df['Zone_norm'].tolist()
sensor_sites_df['OpeningCost_int'] = sensor_sites_df['OpeningCost'].astype(int)
site_costs = dict(zip(sensor_sites_df['Center_norm'], sensor_sites_df['OpeningCost_int']))

def parse_covered_zones(s):
    return set([z.strip() for z in s.split(';') if z.strip() != ''])
site_covers = {}
for (idx, row) in sensor_sites_df.iterrows():
    site = row['Center_norm']
    covered_raw = row['CoveredDistricts']
    covered_zones = parse_covered_zones(covered_raw)
    site_covers[site] = covered_zones
zone_covered_by_sites = {zone: set() for zone in zone_ids}
for site in site_ids:
    for zone in site_covers[site]:
        if zone in zone_covered_by_sites:
            zone_covered_by_sites[zone].add(site)
for zone in zone_ids:
    if len(zone_covered_by_sites[zone]) == 0:
        raise ValueError(f"Monitoring zone '{zone}' is not covered by any sensor site.")
m = gp.Model('MinimumCostSensorCover')
y_vars = m.addVars(site_ids, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((site_costs[site] * y_vars[site] for site in site_ids)), gp.GRB.MINIMIZE)
for zone in zone_ids:
    covering_sites = zone_covered_by_sites[zone]
    m.addConstr(gp.quicksum((y_vars[site] for site in covering_sites)) >= 1, name=f'cover_{zone}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f}')
    print('--- Selected Sensor Sites ---')
    for site in site_ids:
        if y_vars[site].X > 0.5:
            print(f'  Site {site}: INSTALLED (Cost: {site_costs[site]})')
else:
    print(f'No optimal solution found. Status: {m.status}')