import gurobipy as gp
import pandas as pd
import numpy as np
import re

def solve_problem():
    facility_sites_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant14/inputs/facility_sites.csv'
    service_zones_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant14/inputs/service_zones.csv'
    df_sites = pd.read_csv(facility_sites_path, sep=',')
    df_zones = pd.read_csv(service_zones_path, sep=',')
    depots = df_sites['Center'].astype(str).str.strip().tolist()
    opening_cost = dict(zip(df_sites['Center'].astype(str).str.strip(), df_sites['OpeningCost']))
    coverage = {}
    for (idx, row) in df_sites.iterrows():
        depot = str(row['Center']).strip()
        covered_str = str(row['CoveredDistricts'])
        covered_zones = [z.strip() for z in covered_str.split(';') if z.strip()]
        coverage[depot] = set(covered_zones)
    zones = df_zones['Zone'].astype(str).str.strip().tolist()
    zone_covered_by = {z: [] for z in zones}
    for (depot, covered_set) in coverage.items():
        for z in covered_set:
            if z in zone_covered_by:
                zone_covered_by[z].append(depot)
    for z in zones:
        if len(zone_covered_by[z]) == 0:
            raise ValueError(f"Service zone '{z}' is not covered by any depot.")
    m = gp.Model('SetCovering')
    m.Params.MIPGap = 0.0001
    y = m.addVars(depots, vtype=gp.GRB.BINARY, name='')
    m.setObjective(gp.quicksum((opening_cost[d] * y[d] for d in depots)), gp.GRB.MINIMIZE)
    for z in zones:
        depots_covering_z = zone_covered_by[z]
        m.addConstr(gp.quicksum((y[d] for d in depots_covering_z)) >= 1, name=f'cover_{z}')
    m.optimize()
    if m.status == gp.GRB.OPTIMAL:
        print(f'ObjVal {m.objVal}')
        for d in depots:
            print(f'{y[d].VarName} {y[d].X}')
    else:
        print(f'Solver status: {m.status}')
    return m
m = solve_problem()