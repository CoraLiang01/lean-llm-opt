import gurobipy as gp
import pandas as pd
import numpy as np
import re
service_centers_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant4/inputs/service_centers.csv'
service_centers_df = pd.read_csv(service_centers_path, sep=',', dtype=str, keep_default_na=False)
districts_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant4/inputs/districts.csv'
districts_df = pd.read_csv(districts_path, sep=',', dtype=str, keep_default_na=False)
if 'Center' not in service_centers_df.columns:
    raise KeyError("Missing required column 'Center' in service_centers.csv")
centers = service_centers_df['Center'].astype(str).str.strip().tolist()
if 'OpeningCost' not in service_centers_df.columns:
    raise KeyError("Missing required column 'OpeningCost' in service_centers.csv")
try:
    opening_costs = service_centers_df.set_index('Center')['OpeningCost'].astype(float).to_dict()
except Exception as e:
    raise ValueError(f'Failed to convert OpeningCost to float: {e}')
if 'District' not in districts_df.columns:
    raise KeyError("Missing required column 'District' in districts.csv")
districts = districts_df['District'].astype(str).str.strip().tolist()
if 'CoveredDistricts' not in service_centers_df.columns:
    raise KeyError("Missing required column 'CoveredDistricts' in service_centers.csv")
center_covers = {}
for idx, row in service_centers_df.iterrows():
    center_id = str(row['Center']).strip()
    covered_raw = str(row['CoveredDistricts'])
    covered_list = [d.strip() for d in covered_raw.split(';') if d.strip()]
    center_covers[center_id] = set(covered_list)
district_covered_by = {d: set() for d in districts}
for c in centers:
    for d in center_covers[c]:
        if d in district_covered_by:
            district_covered_by[d].add(c)
for d in districts:
    if not district_covered_by[d]:
        raise ValueError(f"District '{d}' is not covered by any center.")
m = gp.Model('EmergencyServiceCenterSetCover')
y_vars = m.addVars(centers, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((opening_costs[c] * y_vars[c] for c in centers)), gp.GRB.MINIMIZE)
for d in districts:
    m.addConstr(gp.quicksum((y_vars[c] for c in district_covered_by[d])) >= 1, name=f'cover_{d}')
m.optimize()