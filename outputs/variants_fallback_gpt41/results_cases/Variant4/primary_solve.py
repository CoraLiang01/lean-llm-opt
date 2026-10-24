import gurobipy as gp
import pandas as pd
import numpy as np
import re
service_centers_path = 'benchmark_dataset/Variant4/inputs/service_centers.csv'
service_centers_df = pd.read_csv(service_centers_path, sep=',', dtype=str, keep_default_na=False)
districts_path = 'benchmark_dataset/Variant4/inputs/districts.csv'
districts_df = pd.read_csv(districts_path, sep=',', dtype=str, keep_default_na=False)
if 'Center' not in service_centers_df.columns:
    raise KeyError("Missing required column 'Center' in service_centers.csv")
centers = service_centers_df['Center'].tolist()
if 'OpeningCost' not in service_centers_df.columns:
    raise KeyError("Missing required column 'OpeningCost' in service_centers.csv")
try:
    opening_costs = service_centers_df.set_index('Center')['OpeningCost'].astype(float).to_dict()
except Exception as e:
    raise ValueError(f'Failed to convert OpeningCost to float: {e}')
if 'CoveredDistricts' not in service_centers_df.columns:
    raise KeyError("Missing required column 'CoveredDistricts' in service_centers.csv")

def parse_districts(cell):
    return [d.strip() for d in cell.split(';') if d.strip()]
center_to_districts = {}
for (idx, row) in service_centers_df.iterrows():
    center = row['Center']
    covered = parse_districts(row['CoveredDistricts'])
    center_to_districts[center] = set(covered)
if 'District' not in districts_df.columns:
    raise KeyError("Missing required column 'District' in districts.csv")
districts = districts_df['District'].tolist()
district_to_centers = {d: set() for d in districts}
for (center, covered_set) in center_to_districts.items():
    for d in covered_set:
        if d in district_to_centers:
            district_to_centers[d].add(center)
for d in districts:
    if not district_to_centers[d]:
        raise ValueError(f"District '{d}' is not covered by any center.")
m = gp.Model('SetCoveringEmergencyCenters')
y_vars = m.addVars(centers, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((opening_costs[i] * y_vars[i] for i in centers)), gp.GRB.MINIMIZE)
for d in districts:
    covering_centers = district_to_centers[d]
    m.addConstr(gp.quicksum((y_vars[i] for i in covering_centers)) >= 1, name=f'cover_{d}')
m.optimize()