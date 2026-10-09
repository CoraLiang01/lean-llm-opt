import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    facility_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP11/facility_costs.csv'
    shipping_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP11/shipping_costs.csv'
    demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP11/demand_requirements.csv'

    def read_csv_multi_enc(path):
        encodings = ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']
        for enc in encodings:
            try:
                return pd.read_csv(path, dtype=str, keep_default_na=False, encoding=enc)
            except Exception:
                continue
        raise RuntimeError(f'Could not read {path} with tried encodings.')
    facility_df = read_csv_multi_enc(facility_path)
    shipping_df = read_csv_multi_enc(shipping_path)
    demand_df = read_csv_multi_enc(demand_path)
    facilities = []
    facility_fixed_cost = {}
    facility_capacity = {}
    for (idx, row) in facility_df.iterrows():
        fac = row['Facility']
        facilities.append(fac)
        try:
            facility_fixed_cost[fac] = float(row['FixedCost'])
        except Exception:
            raise ValueError(f'Invalid FixedCost for facility {fac}')
        try:
            facility_capacity[fac] = float(row['Capacity'])
        except Exception:
            raise ValueError(f'Invalid Capacity for facility {fac}')
    distribution_centers = []
    demand = {}
    for (idx, row) in demand_df.iterrows():
        dc = row['Destination']
        distribution_centers.append(dc)
        try:
            demand[dc] = float(row['Demand'])
        except Exception:
            raise ValueError(f'Invalid Demand for distribution center {dc}')
    shipping_cost = {}
    for (idx, row) in shipping_df.iterrows():
        fac = row['Origin']
        if fac not in facilities:
            continue
        shipping_cost[fac] = {}
        for dc in distribution_centers:
            if dc not in row:
                raise ValueError(f'Shipping cost missing for {fac} to {dc}')
            try:
                shipping_cost[fac][dc] = float(row[dc])
            except Exception:
                raise ValueError(f'Invalid shipping cost for {fac} to {dc}')
    for fac in facilities:
        if fac not in facility_fixed_cost or fac not in facility_capacity:
            raise ValueError(f'Missing facility data for {fac}')
        if fac not in shipping_cost:
            raise ValueError(f'Missing shipping cost row for {fac}')
        for dc in distribution_centers:
            if dc not in shipping_cost[fac]:
                raise ValueError(f'Missing shipping cost for {fac} to {dc}')
    for dc in distribution_centers:
        if dc not in demand:
            raise ValueError(f'Missing demand for {dc}')
    m = gp.Model('ElectroTech_Facility_Location')
    y_vars = m.addVars(facilities, vtype=GRB.BINARY, name='')
    x_keys = [(fac, dc) for fac in facilities for dc in distribution_centers]
    x_vars = m.addVars(x_keys, lb=0, vtype=GRB.CONTINUOUS, name='')
    m.setObjective(gp.quicksum((facility_fixed_cost[fac] * y_vars[fac] for fac in facilities)) + gp.quicksum((shipping_cost[fac][dc] * x_vars[fac, dc] for fac in facilities for dc in distribution_centers)), GRB.MINIMIZE)
    m.addConstrs((gp.quicksum((x_vars[fac, dc] for fac in facilities)) == demand[dc] for dc in distribution_centers), name='')
    m.addConstrs((gp.quicksum((x_vars[fac, dc] for dc in distribution_centers)) <= facility_capacity[fac] * y_vars[fac] for fac in facilities), name='')
    m.Params.MIPGap = 0.0001
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()