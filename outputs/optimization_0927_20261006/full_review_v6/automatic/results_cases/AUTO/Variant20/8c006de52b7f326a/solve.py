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
if 'OpeningCost' not in sensor_sites_df.columns:
    raise KeyError("Missing required column 'OpeningCost' in sensor_sites.csv")
opening_cost = {}
for (idx, row) in sensor_sites_df.iterrows():
    site = row['Center']
    try:
        cost = int(row['OpeningCost'])
    except Exception as e:
        raise ValueError(f"Invalid OpeningCost for site {site}: {row['OpeningCost']}")
    opening_cost[site] = cost
site_covers = {}
for (idx, row) in sensor_sites_df.iterrows():
    site = row['Center']
    covered_field = row['CoveredDistricts']
    covered_zones = [z.strip() for z in covered_field.split(';') if z.strip() != '']
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
    covering_sites = zone_covered_by[zone]
    m.addConstr(gp.quicksum((y_vars[site] for site in covering_sites)) >= 1, name=f'cover_{zone}')
m.optimize()