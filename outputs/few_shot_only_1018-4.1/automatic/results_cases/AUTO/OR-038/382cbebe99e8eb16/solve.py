import pandas as pd
import gurobipy as gp
from gurobipy import GRB
capacity_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA4/NewCarSalesInNorway1/capacity.csv', dtype=str, keep_default_na=False)
capacity_df['Capacity'] = capacity_df['Capacity'].astype(int)
vehicle_types = list(capacity_df['VehicleType'])
capacity = dict(zip(capacity_df['VehicleType'], capacity_df['Capacity']))
products_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA4/NewCarSalesInNorway1/products.csv', dtype=str, keep_default_na=False)
products_df['Value'] = products_df['Value'].astype(int)
value = dict(zip(products_df['ProductName'], products_df['Value']))
missing_value = [v for v in vehicle_types if v not in value]
missing_capacity = [v for v in vehicle_types if v not in capacity]
if missing_value or missing_capacity:
    raise ValueError(f'Missing value for: {missing_value}, missing capacity for: {missing_capacity}')
total_capacity = sum((capacity[v] for v in vehicle_types))
m = gp.Model('NewCarSalesInNorway1')
x_vars = m.addVars(vehicle_types, lb=0, ub={v: capacity[v] for v in vehicle_types}, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((value[v] * x_vars[v] for v in vehicle_types)), GRB.MAXIMIZE)
m.addConstr(gp.quicksum((x_vars[v] for v in vehicle_types)) <= total_capacity, name='total_capacity')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in vehicle_types:
        print(f'{x_vars[v].VarName}: {x_vars[v].X}')
else:
    print(f'Solver status: {m.Status}')