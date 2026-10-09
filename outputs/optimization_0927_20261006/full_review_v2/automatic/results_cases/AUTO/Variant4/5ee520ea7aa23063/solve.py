import gurobipy as gp
import pandas as pd
import numpy as np
import re
service_centers_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant4/inputs/service_centers.csv', dtype=str, keep_default_na=False)
districts_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant4/inputs/districts.csv', dtype=str, keep_default_na=False)
service_centers_df['Center_norm'] = service_centers_df['Center'].str.strip().str.casefold()
districts_df['District_norm'] = districts_df['District'].str.strip().str.casefold()
centers = service_centers_df['Center'].tolist()
centers_norm = service_centers_df['Center_norm'].tolist()
districts = districts_df['District'].tolist()
districts_norm = districts_df['District_norm'].tolist()
center_norm_to_id = dict(zip(centers_norm, centers))
district_norm_to_id = dict(zip(districts_norm, districts))
try:
    opening_cost = {row['Center']: int(row['OpeningCost']) for (_, row) in service_centers_df.iterrows()}
except Exception as e:
    raise ValueError(f'Failed to parse OpeningCost as int for all centers: {e}')
center_to_covered_districts_norm = {}
for (_, row) in service_centers_df.iterrows():
    center_id = row['Center']
    covered = [d.strip().casefold() for d in re.split(';', row['CoveredDistricts']) if d.strip() != '']
    center_to_covered_districts_norm[center_id] = set(covered)
district_to_covering_centers = {district: set() for district in districts}
for center_id in centers:
    covered_norms = center_to_covered_districts_norm[center_id]
    for (d_idx, district_norm) in enumerate(districts_norm):
        if district_norm in covered_norms:
            district_to_covering_centers[districts[d_idx]].add(center_id)
for district in districts:
    if not district_to_covering_centers[district]:
        raise ValueError(f"District '{district}' is not covered by any center.")
m = gp.Model('EmergencyServiceCenterSetCover')
y_vars = m.addVars(centers, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((opening_cost[center] * y_vars[center] for center in centers)), gp.GRB.MINIMIZE)
for district in districts:
    covering_centers = district_to_covering_centers[district]
    m.addConstr(gp.quicksum((y_vars[center] for center in covering_centers)) >= 1, name=f'cover_{district}')
m.optimize()