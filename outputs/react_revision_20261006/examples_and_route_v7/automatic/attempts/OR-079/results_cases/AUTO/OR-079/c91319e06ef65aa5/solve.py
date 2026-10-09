import gurobipy as gp
import pandas as pd
import numpy as np
facility_costs_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP11/facility_costs.csv', dtype=str, keep_default_na=False)
shipping_costs_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP11/shipping_costs.csv', dtype=str, keep_default_na=False)
demand_requirements_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP11/demand_requirements.csv', dtype=str, keep_default_na=False)
facilities = facility_costs_df['Facility'].tolist()
distribution_centers = demand_requirements_df['Destination'].tolist()
facility_fixed_cost = {}
facility_capacity = {}
for (_, row) in facility_costs_df.iterrows():
    fac = row['Facility']
    if fac not in facilities:
        continue
    try:
        facility_fixed_cost[fac] = int(row['FixedCost'])
        facility_capacity[fac] = int(row['Capacity'])
    except Exception as e:
        raise ValueError(f'Invalid numeric value in facility_costs.csv for facility {fac}: {e}')
demand = {}
for (_, row) in demand_requirements_df.iterrows():
    dc = row['Destination']
    if dc not in distribution_centers:
        continue
    try:
        demand[dc] = int(row['Demand'])
    except Exception as e:
        raise ValueError(f'Invalid numeric value in demand_requirements.csv for destination {dc}: {e}')
shipping_cost = {}
shipping_facilities = shipping_costs_df['Origin'].tolist()
shipping_dcs = [col for col in shipping_costs_df.columns if col != 'Origin']
missing_facilities = set(facilities) - set(shipping_facilities)
if missing_facilities:
    raise ValueError(f'Missing facilities in shipping_costs.csv: {missing_facilities}')
missing_dcs = set(distribution_centers) - set(shipping_dcs)
if missing_dcs:
    raise ValueError(f'Missing distribution centers in shipping_costs.csv: {missing_dcs}')
for (_, row) in shipping_costs_df.iterrows():
    fac = row['Origin']
    if fac not in facilities:
        continue
    for dc in distribution_centers:
        try:
            shipping_cost[fac, dc] = int(row[dc])
        except Exception as e:
            raise ValueError(f'Invalid shipping cost for ({fac}, {dc}): {e}')
for fac in facilities:
    if fac not in facility_fixed_cost or fac not in facility_capacity:
        raise ValueError(f'Missing fixed cost or capacity for facility {fac}')
for dc in distribution_centers:
    if dc not in demand:
        raise ValueError(f'Missing demand for distribution center {dc}')
for fac in facilities:
    for dc in distribution_centers:
        if (fac, dc) not in shipping_cost:
            raise ValueError(f'Missing shipping cost for facility {fac} to DC {dc}')
m = gp.Model('UFLP')
y_vars = m.addVars(facilities, vtype=gp.GRB.BINARY, name='')
x_keys = [(fac, dc) for fac in facilities for dc in distribution_centers]
x_vars = m.addVars(x_keys, vtype=gp.GRB.CONTINUOUS, lb=0, name='')
m.setObjective(gp.quicksum((facility_fixed_cost[fac] * y_vars[fac] for fac in facilities)) + gp.quicksum((shipping_cost[fac, dc] * x_vars[fac, dc] for fac in facilities for dc in distribution_centers)), gp.GRB.MINIMIZE)
for dc in distribution_centers:
    m.addConstr(gp.quicksum((x_vars[fac, dc] for fac in facilities)) == demand[dc], name=f'demand_{dc}')
for fac in facilities:
    m.addConstr(gp.quicksum((x_vars[fac, dc] for dc in distribution_centers)) <= facility_capacity[fac] * y_vars[fac], name=f'capacity_{fac}')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for fac in facilities:
        print(f'{y_vars[fac].VarName} {y_vars[fac].X}')
    for (fac, dc) in x_keys:
        print(f'{x_vars[fac, dc].VarName} {x_vars[fac, dc].X}')
else:
    print(f'Solver status: {m.Status}')