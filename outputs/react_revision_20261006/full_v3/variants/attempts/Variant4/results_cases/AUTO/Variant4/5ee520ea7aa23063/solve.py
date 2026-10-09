import gurobipy as gp
import pandas as pd
import numpy as np
import re
service_centers_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant4/inputs/service_centers.csv'
districts_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant4/inputs/districts.csv'
df_centers = pd.read_csv(service_centers_path, sep=',')
df_districts = pd.read_csv(districts_path, sep=',')
df_centers['Center'] = df_centers['Center'].astype(str).str.strip()
df_centers['CoveredDistricts'] = df_centers['CoveredDistricts'].astype(str).str.strip()
df_districts['District'] = df_districts['District'].astype(str).str.strip()
centers = df_centers['Center'].tolist()
districts = df_districts['District'].tolist()
opening_cost = df_centers.set_index('Center')['OpeningCost'].to_dict()
center_covers = {}
district_covered_by = {d: set() for d in districts}
for (idx, row) in df_centers.iterrows():
    center = row['Center']
    covered_str = row['CoveredDistricts']
    covered_list = [d.strip() for d in covered_str.split(';') if d.strip()]
    center_covers[center] = set(covered_list)
    for d in covered_list:
        if d in district_covered_by:
            district_covered_by[d].add(center)
for d in districts:
    if len(district_covered_by[d]) == 0:
        raise ValueError(f"District '{d}' is not covered by any center.")
m = gp.Model('SetCoveringEmergencyCenters')
m.Params.MIPGap = 0.0001
y = m.addVars(centers, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((opening_cost[i] * y[i] for i in centers)), gp.GRB.MINIMIZE)
for d in districts:
    m.addConstr(gp.quicksum((y[i] for i in district_covered_by[d])) >= 1, name=f'cover_{d}')
m.optimize()
if m.Status == gp.GRB.OPTIMAL:
    print(f'ObjVal {m.ObjVal}')
    for i in centers:
        print(f'{y[i].VarName} {y[i].X}')
else:
    print(f'Solver status: {m.Status}')