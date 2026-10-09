import gurobipy as gp
import pandas as pd
import numpy as np
import re
facilities_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant14/inputs/facility_sites.csv', sep=',')
zones_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant14/inputs/service_zones.csv', sep=',')

def norm_id(x):
    return str(x).strip()
depots = [norm_id(c) for c in facilities_df['Center']]
zones = [norm_id(z) for z in zones_df['Zone']]
opening_cost = {}
for (idx, row) in facilities_df.iterrows():
    depot = norm_id(row['Center'])
    cost = row['OpeningCost']
    opening_cost[depot] = float(cost)
depot_covers = {}
zone_covered_by = {z: set() for z in zones}
for (idx, row) in facilities_df.iterrows():
    depot = norm_id(row['Center'])
    covered_str = str(row['CoveredDistricts'])
    covered_zones = [norm_id(z) for z in covered_str.split(';') if z.strip()]
    depot_covers[depot] = set(covered_zones)
    for z in covered_zones:
        if z in zone_covered_by:
            zone_covered_by[z].add(depot)
        else:
            raise ValueError(f"Depot {depot} covers unknown zone '{z}' not in service_zones.csv.")
for z in zones:
    if len(zone_covered_by[z]) == 0:
        raise ValueError(f"Service zone '{z}' is not covered by any depot.")
m = gp.Model('SetCoveringDepots')
y = m.addVars(depots, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((opening_cost[d] * y[d] for d in depots)), gp.GRB.MINIMIZE)
for z in zones:
    covering_depots = zone_covered_by[z]
    m.addConstr(gp.quicksum((y[d] for d in covering_depots)) >= 1, name=f'cover_{z}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total opening cost: {m.objVal:.2f}')
    print('Opened depots:')
    for d in depots:
        if y[d].X > 0.5:
            print(f'  Depot {d} (Opening cost: {opening_cost[d]:.2f})')
else:
    print(f'No optimal solution found. Status: {m.status}')