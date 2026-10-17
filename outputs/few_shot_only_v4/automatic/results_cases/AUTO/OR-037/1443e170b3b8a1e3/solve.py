import pandas as pd
import gurobipy as gp
from gurobipy import GRB
products_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA3/CarSales2/products.csv', dtype=str, keep_default_na=False)
capacity_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA3/CarSales2/capacity.csv', dtype=str, keep_default_na=False)
vehicle_types = products_df['ProductName'].tolist()
try:
    profit = {row['ProductName']: int(row['p_i']) for _, row in products_df.iterrows()}
    weight = {row['ProductName']: int(row['w_i']) for _, row in products_df.iterrows()}
except Exception as e:
    raise ValueError(f'Error parsing products.csv: {e}')
if 'C' in capacity_df.columns:
    try:
        C = int(capacity_df['C'].iloc[0])
    except Exception as e:
        raise ValueError(f'Error parsing capacity.csv: {e}')
else:
    try:
        C = int(capacity_df.iloc[0, 0])
    except Exception as e:
        raise ValueError(f'Error parsing capacity.csv: {e}')
if set(profit.keys()) != set(vehicle_types) or set(weight.keys()) != set(vehicle_types):
    raise ValueError('Mismatch in vehicle types and profit/weight keys.')
m = gp.Model('CarSales_Inventory_Optimization')
m.Params.MIPGap = 0.0001
x_vars = m.addVars(vehicle_types, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((profit[i] * x_vars[i] for i in vehicle_types)), GRB.MAXIMIZE)
m.addConstr(gp.quicksum((weight[i] * x_vars[i] for i in vehicle_types)) <= C, name='capacity')
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')