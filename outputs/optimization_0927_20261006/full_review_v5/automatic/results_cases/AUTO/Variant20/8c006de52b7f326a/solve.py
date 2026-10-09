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

def norm_id(x):
    return x.strip().casefold()
norm_zone_to_orig = {norm_id(z): z for z in monitoring_zones}
site_covers = {}
for (_, row) in sensor_sites_df.iterrows():
    site = row['Center']
    covered_raw = row['CoveredDistricts']
    covered_zones = [z.strip() for z in covered_raw.split(';') if z.strip() != '']
    covered_zones_norm = set((norm_id(z) for z in covered_zones))
    site_covers[site] = covered_zones_norm
zone_covered_by = {z: set() for z in monitoring_zones}
for site in sensor_sites:
    for norm_zone in site_covers[site]:
        if norm_zone in norm_zone_to_orig:
            orig_zone = norm_zone_to_orig[norm_zone]
            zone_covered_by[orig_zone].add(site)
for z in monitoring_zones:
    if len(zone_covered_by[z]) == 0:
        raise ValueError(f"Monitoring zone '{z}' is not covered by any sensor site.")
m = gp.Model('MinimumCostSensorCover')
y_vars = m.addVars(sensor_sites, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((opening_cost[i] * y_vars[i] for i in sensor_sites)), gp.GRB.MINIMIZE)
for z in monitoring_zones:
    m.addConstr(gp.quicksum((y_vars[i] for i in zone_covered_by[z])) >= 1, name=f'cover_{z}')
m.optimize()