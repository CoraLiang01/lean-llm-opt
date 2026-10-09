import gurobipy as gp
import pandas as pd
import numpy as np
import re
facility_sites_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant14/inputs/facility_sites.csv'
service_zones_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant14/inputs/service_zones.csv'
facility_sites_df = pd.read_csv(facility_sites_path, sep=',', dtype=str, keep_default_na=False)
service_zones_df = pd.read_csv(service_zones_path, sep=',', dtype=str, keep_default_na=False)

def norm_id(x):
    return x.strip().casefold()
depots = facility_sites_df['Center'].apply(norm_id).tolist()
zones = service_zones_df['Zone'].apply(norm_id).tolist()
if facility_sites_df['OpeningCost'].isnull().any():
    raise ValueError('Missing OpeningCost for some depots.')
try:
    opening_costs = {norm_id(row['Center']): int(row['OpeningCost']) for (_, row) in facility_sites_df.iterrows()}
except Exception as e:
    raise ValueError(f'Error converting OpeningCost to int: {e}')
coverage = {}
for (_, row) in facility_sites_df.iterrows():
    depot_id = norm_id(row['Center'])
    covered_str = row['CoveredDistricts']
    covered_zones = [norm_id(z) for z in covered_str.split(';') if z.strip() != '']
    coverage[depot_id] = set(covered_zones)
zone_to_depots = {z: set() for z in zones}
for (depot_id, covered_set) in coverage.items():
    for z in covered_set:
        if z in zone_to_depots:
            zone_to_depots[z].add(depot_id)
for z in zones:
    if not zone_to_depots[z]:
        raise ValueError(f"Service zone '{z}' is not covered by any depot.")

def solve_problem(depots, zones, opening_costs, zone_to_depots):
    m = gp.Model('SetCovering')
    y_vars = m.addVars(depots, vtype=gp.GRB.BINARY, name='')
    m.setObjective(gp.quicksum((opening_costs[i] * y_vars[i] for i in depots)), gp.GRB.MINIMIZE)
    for z in zones:
        m.addConstr(gp.quicksum((y_vars[i] for i in zone_to_depots[z])) >= 1, name='cov')
    m.Params.MIPGap = 0.0001
    m.optimize()
    return m
m = solve_problem(depots, zones, opening_costs, zone_to_depots)
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')