import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    facility_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP11/facility_costs.csv'
    shipping_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP11/shipping_costs.csv'
    demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP11/demand_requirements.csv'
    encodings = ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']
    for enc in encodings:
        try:
            facility_df = pd.read_csv(facility_path, encoding=enc)
            break
        except Exception:
            continue
    else:
        raise RuntimeError(f'Could not read {facility_path} with tried encodings.')
    for enc in encodings:
        try:
            shipping_df = pd.read_csv(shipping_path, encoding=enc)
            break
        except Exception:
            continue
    else:
        raise RuntimeError(f'Could not read {shipping_path} with tried encodings.')
    for enc in encodings:
        try:
            demand_df = pd.read_csv(demand_path, encoding=enc)
            break
        except Exception:
            continue
    else:
        raise RuntimeError(f'Could not read {demand_path} with tried encodings.')
    facilities = facility_df['Facility'].astype(str).tolist()
    distribution_centers = demand_df['Destination'].astype(str).tolist()
    fixed_cost = {}
    capacity = {}
    for (_, row) in facility_df.iterrows():
        key = str(row['Facility'])
        fixed_cost[key] = float(row['FixedCost'])
        capacity[key] = float(row['Capacity'])
    demand = {}
    for (_, row) in demand_df.iterrows():
        key = str(row['Destination'])
        demand[key] = float(row['Demand'])
    shipping_cost = {}
    for (_, row) in shipping_df.iterrows():
        origin = str(row['Origin'])
        shipping_cost[origin] = {}
        for dc in distribution_centers:
            if dc not in row:
                raise ValueError(f'Distribution center {dc} missing in shipping_costs.csv columns.')
            shipping_cost[origin][dc] = float(row[dc])
    if set(facilities) != set(shipping_cost.keys()):
        raise ValueError('Mismatch between facilities in facility_costs.csv and shipping_costs.csv.')
    for i in facilities:
        if set(distribution_centers) != set(shipping_cost[i].keys()):
            raise ValueError(f'Mismatch in shipping cost columns for facility {i}.')
    x_keys = [(i, j) for i in facilities for j in distribution_centers]
    y_keys = facilities
    m = gp.Model('ElectroTech_UFLP')
    x = m.addVars(x_keys, lb=0, vtype=GRB.CONTINUOUS, name='')
    y = m.addVars(y_keys, vtype=GRB.BINARY, name='')
    m.setObjective(gp.quicksum((fixed_cost[i] * y[i] for i in facilities)) + gp.quicksum((shipping_cost[i][j] * x[i, j] for (i, j) in x_keys)), GRB.MINIMIZE)
    m.addConstrs((gp.quicksum((x[i, j] for i in facilities)) == demand[j] for j in distribution_centers), name='')
    m.addConstrs((gp.quicksum((x[i, j] for j in distribution_centers)) <= capacity[i] * y[i] for i in facilities), name='')
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