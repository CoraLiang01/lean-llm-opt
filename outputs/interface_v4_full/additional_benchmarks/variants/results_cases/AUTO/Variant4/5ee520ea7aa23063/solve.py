import gurobipy as gp
import pandas as pd
import numpy as np
import re

def solve_set_covering():
    centers_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant4/inputs/service_centers.csv', sep=',')
    districts_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant4/inputs/districts.csv', sep=',')
    centers = centers_df['Center'].astype(str).tolist()
    districts = districts_df['District'].astype(str).tolist()
    opening_cost = dict(zip(centers_df['Center'].astype(str), centers_df['OpeningCost']))
    center_covers = {}
    for idx, row in centers_df.iterrows():
        center = str(row['Center'])
        covered = [d.strip() for d in str(row['CoveredDistricts']).split(';') if d.strip()]
        center_covers[center] = set(covered)
    district_to_centers = {d: set() for d in districts}
    for center, covered_set in center_covers.items():
        for d in covered_set:
            if d in district_to_centers:
                district_to_centers[d].add(center)
    uncovered = [d for d, cset in district_to_centers.items() if not cset]
    if uncovered:
        raise ValueError(f'The following districts are not covered by any center: {uncovered}')
    m = gp.Model('EmergencyServiceCenterSetCovering')
    y = m.addVars(centers, vtype=gp.GRB.BINARY, name='')
    m.setObjective(gp.quicksum((opening_cost[i] * y[i] for i in centers)), gp.GRB.MINIMIZE)
    for d in districts:
        m.addConstr(gp.quicksum((y[i] for i in district_to_centers[d])) >= 1, name=f'cover_{d}')
    m.optimize()
    return m
m = solve_set_covering()