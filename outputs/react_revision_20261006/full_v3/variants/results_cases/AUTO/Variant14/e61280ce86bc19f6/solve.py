import gurobipy as gp
import pandas as pd
import numpy as np
import re
facility_sites_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant14/inputs/facility_sites.csv'
service_zones_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant14/inputs/service_zones.csv'
fac_df = pd.read_csv(facility_sites_path, sep=',')
zones_df = pd.read_csv(service_zones_path, sep=',')

def norm_id(x):
    return str(x).strip().casefold()
depots = fac_df['Center'].astype(str).tolist()
zones = zones_df['Zone'].astype(str).tolist()
if fac_df['Center'].isnull().any() or fac_df['OpeningCost'].isnull().any():
    raise ValueError('Missing Center or OpeningCost in facility_sites.csv')
opening_cost = dict(zip(fac_df['Center'].astype(str), fac_df['OpeningCost']))
coverage = {}
for (idx, row) in fac_df.iterrows():
    depot = str(row['Center'])
    covered_str = str(row['CoveredDistricts'])
    covered_zones = [z.strip() for z in covered_str.split(';') if z.strip()]
    coverage[depot] = set(covered_zones)
zone_covered_by = {z: [] for z in zones}
for (depot, covered_set) in coverage.items():
    for z in covered_set:
        if z in zone_covered_by:
            zone_covered_by[z].append(depot)
uncovered = [z for (z, depots_covering) in zone_covered_by.items() if len(depots_covering) == 0]
if uncovered:
    raise ValueError(f'The following service zones are not covered by any depot: {uncovered}')
m = gp.Model('SetCoveringFacilityLocation')
m.Params.MIPGap = 0.0001
y = m.addVars(depots, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((opening_cost[i] * y[i] for i in depots)), gp.GRB.MINIMIZE)
for z in zones:
    covering_depots = zone_covered_by[z]
    m.addConstr(gp.quicksum((y[i] for i in covering_depots)) >= 1, name=f'cover_{z}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for i in depots:
        print(f'{y[i].VarName} {y[i].X}')
else:
    print(f'Solver status: {m.status}')