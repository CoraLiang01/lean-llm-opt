import gurobipy as gp
import pandas as pd
import numpy as np
import re
sensor_sites_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant20/inputs/sensor_sites.csv'
sensor_sites_df = pd.read_csv(sensor_sites_path, sep=',', dtype=str, keep_default_na=False)
monitoring_zones_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant20/inputs/monitoring_zones.csv'
monitoring_zones_df = pd.read_csv(monitoring_zones_path, sep=',', dtype=str, keep_default_na=False)
sensor_sites_df['Center'] = sensor_sites_df['Center'].str.strip()
sensor_sites_df['OpeningCost'] = sensor_sites_df['OpeningCost'].astype(int)
sensor_sites_df['CoveredDistricts'] = sensor_sites_df['CoveredDistricts'].str.strip()
monitoring_zones_df['Zone'] = monitoring_zones_df['Zone'].str.strip()
site_ids = list(sensor_sites_df['Center'])
zone_ids = list(monitoring_zones_df['Zone'])
site_to_zones = {}
for (idx, row) in sensor_sites_df.iterrows():
    site = row['Center']
    covered = [z.strip() for z in row['CoveredDistricts'].split(';') if z.strip() != '']
    site_to_zones[site] = set(covered)
zone_to_sites = {zone: set() for zone in zone_ids}
for (site, covered_zones) in site_to_zones.items():
    for zone in covered_zones:
        if zone in zone_to_sites:
            zone_to_sites[zone].add(site)
uncovered_zones = [zone for (zone, sites) in zone_to_sites.items() if len(sites) == 0]
if uncovered_zones:
    raise ValueError(f'The following zones are not covered by any site: {uncovered_zones}')
opening_cost = {row['Center']: row['OpeningCost'] for (idx, row) in sensor_sites_df.iterrows()}
m = gp.Model('SensorSetCover')
y_vars = m.addVars(site_ids, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((opening_cost[site] * y_vars[site] for site in site_ids)), gp.GRB.MINIMIZE)
for zone in zone_ids:
    covering_sites = zone_to_sites[zone]
    m.addConstr(gp.quicksum((y_vars[site] for site in covering_sites)) >= 1, name=f'cover_{zone}')
m.setParam('MIPGap', 0.0001)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for site in site_ids:
        print(f'{y_vars[site].VarName} {y_vars[site].X}')
else:
    print(f'Solver status: {m.status}')