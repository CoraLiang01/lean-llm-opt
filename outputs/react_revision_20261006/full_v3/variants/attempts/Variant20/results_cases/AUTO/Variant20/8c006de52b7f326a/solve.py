import gurobipy as gp
import pandas as pd
import numpy as np
import re
sensor_sites_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant20/inputs/sensor_sites.csv'
zones_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant20/inputs/monitoring_zones.csv'
sensor_sites_df = pd.read_csv(sensor_sites_path, sep=',')
sensor_sites_df['Center'] = sensor_sites_df['Center'].astype(str).str.strip()
sensor_sites_df['OpeningCost'] = sensor_sites_df['OpeningCost'].astype(int)

def parse_covered_districts(s):
    return [z.strip() for z in str(s).split(';') if z.strip()]
sensor_sites_df['CoveredDistrictsList'] = sensor_sites_df['CoveredDistricts'].apply(parse_covered_districts)
zones_df = pd.read_csv(zones_path, sep=',')
zones_df['Zone'] = zones_df['Zone'].astype(str).str.strip()
sensor_sites = list(sensor_sites_df['Center'].unique())
zones = list(zones_df['Zone'].unique())
covered_zone_set = set()
for lst in sensor_sites_df['CoveredDistrictsList']:
    covered_zone_set.update(lst)
missing_zones = set(zones) - covered_zone_set
if missing_zones:
    raise ValueError(f'The following monitoring zones are not covered by any sensor site: {sorted(missing_zones)}')
zone_to_sites = {z: [] for z in zones}
for (idx, row) in sensor_sites_df.iterrows():
    site = row['Center']
    covered = row['CoveredDistrictsList']
    for z in covered:
        if z in zone_to_sites:
            zone_to_sites[z].append(site)
for z in zones:
    if not zone_to_sites[z]:
        raise ValueError(f'Zone {z} is not covered by any sensor site.')
site_cost = dict(zip(sensor_sites_df['Center'], sensor_sites_df['OpeningCost']))

def solve_sensor_covering(sensor_sites, zones, site_cost, zone_to_sites):
    m = gp.Model('SensorSetCover')
    y = m.addVars(sensor_sites, vtype=gp.GRB.BINARY, name='')
    m.setObjective(gp.quicksum((site_cost[i] * y[i] for i in sensor_sites)), gp.GRB.MINIMIZE)
    for z in zones:
        m.addConstr(gp.quicksum((y[i] for i in zone_to_sites[z])) >= 1, name=f'cov_{z}')
    m.Params.MIPGap = 0.0001
    m.optimize()
    return m
m = solve_sensor_covering(sensor_sites, zones, site_cost, zone_to_sites)
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')