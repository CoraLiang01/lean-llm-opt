import gurobipy as gp
import pandas as pd
import numpy as np
import re
service_centers_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant4/inputs/service_centers.csv'
districts_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant4/inputs/districts.csv'
df_centers = pd.read_csv(service_centers_path, sep=',')
df_districts = pd.read_csv(districts_path, sep=',')
df_centers['Center'] = df_centers['Center'].astype(str).str.strip()
df_districts['District'] = df_districts['District'].astype(str).str.strip()
centers = df_centers['Center'].tolist()
districts = df_districts['District'].tolist()
if len(set(centers)) != len(centers):
    raise ValueError('Duplicate Center identifiers found in service_centers.csv')
if len(set(districts)) != len(districts):
    raise ValueError('Duplicate District identifiers found in districts.csv')
opening_cost = dict(zip(df_centers['Center'], df_centers['OpeningCost']))
center_covers = {}
for (idx, row) in df_centers.iterrows():
    center = row['Center']
    covered_str = str(row['CoveredDistricts']).strip()
    if covered_str == '' or pd.isnull(covered_str):
        covered = []
    else:
        covered = [d.strip() for d in covered_str.split(';') if d.strip() != '']
    center_covers[center] = set(covered)
district_covered_by = {d: set() for d in districts}
for c in centers:
    for d in center_covers[c]:
        if d in district_covered_by:
            district_covered_by[d].add(c)
for d in districts:
    if len(district_covered_by[d]) == 0:
        raise ValueError(f"District '{d}' is not covered by any center.")
m = gp.Model('SetCoveringEmergencyCenters')
y = m.addVars(centers, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((opening_cost[i] * y[i] for i in centers)), gp.GRB.MINIMIZE)
for d in districts:
    m.addConstr(gp.quicksum((y[c] for c in district_covered_by[d])) >= 1, name=f'cover_{d}')
m.Params.MIPGap = 0.0001
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')