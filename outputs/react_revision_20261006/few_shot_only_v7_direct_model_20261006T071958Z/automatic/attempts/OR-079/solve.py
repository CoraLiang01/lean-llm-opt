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
            facility_df = pd.read_csv(facility_path, dtype=str, keep_default_na=False, encoding=enc)
            break
        except Exception:
            continue
    else:
        raise RuntimeError(f'Could not read {facility_path} with tried encodings.')
    for enc in encodings:
        try:
            shipping_df = pd.read_csv(shipping_path, dtype=str, keep_default_na=False, encoding=enc)
            break
        except Exception:
            continue
    else:
        raise RuntimeError(f'Could not read {shipping_path} with tried encodings.')
    for enc in encodings:
        try:
            demand_df = pd.read_csv(demand_path, dtype=str, keep_default_na=False, encoding=enc)
            break
        except Exception:
            continue
    else:
        raise RuntimeError(f'Could not read {demand_path} with tried encodings.')
    I = facility_df['Facility'].tolist()
    J = demand_df['Destination'].tolist()
    if facility_df['Facility'].duplicated().any():
        raise ValueError('Duplicate Facility identifiers in facility_costs.csv')
    fixed_cost = {}
    capacity = {}
    for (_, row) in facility_df.iterrows():
        i = row['Facility']
        try:
            fixed_cost[i] = float(row['FixedCost'])
            capacity[i] = float(row['Capacity'])
        except Exception:
            raise ValueError(f'Non-numeric FixedCost or Capacity for facility {i}')
    if demand_df['Destination'].duplicated().any():
        raise ValueError('Duplicate Destination identifiers in demand_requirements.csv')
    demand = {}
    for (_, row) in demand_df.iterrows():
        j = row['Destination']
        try:
            demand[j] = float(row['Demand'])
        except Exception:
            raise ValueError(f'Non-numeric Demand for destination {j}')
    if shipping_df['Origin'].duplicated().any():
        raise ValueError('Duplicate Origin identifiers in shipping_costs.csv')
    ship_cost = {}
    for (_, row) in shipping_df.iterrows():
        i = row['Origin']
        ship_cost[i] = {}
        for j in J:
            if j not in row:
                raise ValueError(f'Destination {j} missing in shipping_costs.csv columns')
            try:
                ship_cost[i][j] = float(row[j])
            except Exception:
                raise ValueError(f'Non-numeric ShipCost for Origin {i}, Destination {j}')
    for i in I:
        if i not in ship_cost:
            raise ValueError(f'Facility {i} missing in shipping_costs.csv')
        for j in J:
            if j not in ship_cost[i]:
                raise ValueError(f'Destination {j} missing for facility {i} in shipping_costs.csv')
    m = gp.Model('ElectroTech_Facility_Location')
    m.Params.MIPGap = 0.0001
    y_vars = m.addVars(I, vtype=GRB.BINARY, name='')
    x_keys = [(i, j) for i in I for j in J]
    x_vars = m.addVars(x_keys, lb=0, vtype=GRB.CONTINUOUS, name='')
    m.setObjective(gp.quicksum((fixed_cost[i] * y_vars[i] for i in I)) + gp.quicksum((ship_cost[i][j] * x_vars[i, j] for i in I for j in J)), GRB.MINIMIZE)
    m.addConstrs((gp.quicksum((x_vars[i, j] for i in I)) == demand[j] for j in J), name='')
    m.addConstrs((gp.quicksum((x_vars[i, j] for j in J)) <= capacity[i] * y_vars[i] for i in I), name='')
    m.optimize()
    return m
m = solve_problem()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')