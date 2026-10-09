import gurobipy as gp
import pandas as pd
import numpy as np
import re
service_centers_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant4/inputs/service_centers.csv'
districts_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant4/inputs/districts.csv'
centers_df = pd.read_csv(service_centers_path, dtype=str, keep_default_na=False)
districts_df = pd.read_csv(districts_path, dtype=str, keep_default_na=False)
centers = centers_df['Center'].astype(str).tolist()
districts = districts_df['District'].astype(str).tolist()
try:
    opening_cost = {}
    for (idx, row) in centers_df.iterrows():
        center_id = str(row['Center'])
        try:
            cost = int(row['OpeningCost'])
        except Exception:
            raise ValueError(f"OpeningCost for center {center_id} is not a valid integer: {row['OpeningCost']}")
        opening_cost[center_id] = cost
    if set(opening_cost.keys()) != set(centers):
        raise ValueError('Mismatch in centers and opening_cost keys.')
except Exception as e:
    raise RuntimeError(f'Error processing OpeningCost: {e}')
center_covers = {}
for (idx, row) in centers_df.iterrows():
    center_id = str(row['Center'])
    covered_str = str(row['CoveredDistricts'])
    covered_list = [d.strip() for d in covered_str.split(';') if d.strip() != '']
    center_covers[center_id] = set(covered_list)
district_covered_by = {d: set() for d in districts}
for (center_id, covered_set) in center_covers.items():
    for d in covered_set:
        if d in district_covered_by:
            district_covered_by[d].add(center_id)
for d in districts:
    if len(district_covered_by[d]) == 0:
        raise ValueError(f'District {d} is not covered by any center.')

def solve_set_covering(centers, districts, opening_cost, district_covered_by):
    m = gp.Model('SetCovering')
    m.setParam('MIPGap', 0.0001)
    y_vars = m.addVars(centers, vtype=gp.GRB.BINARY, name='')
    m.setObjective(gp.quicksum((opening_cost[i] * y_vars[i] for i in centers)), gp.GRB.MINIMIZE)
    for d in districts:
        m.addConstr(gp.quicksum((y_vars[i] for i in district_covered_by[d])) >= 1, name=f'cover_{d}')
    m.optimize()
    return m
m = solve_set_covering(centers, districts, opening_cost, district_covered_by)
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')