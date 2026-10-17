import gurobipy as gp
from gurobipy import GRB
import pandas as pd
facility_costs_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP11/facility_costs.csv'
facility_df = pd.read_csv(facility_costs_path, dtype=str, keep_default_na=False)
if 'Facility' not in facility_df.columns or 'FixedCost' not in facility_df.columns:
    raise ValueError('facility_costs.csv missing required columns')
facilities = facility_df['Facility'].tolist()
fixed_cost = {}
for idx, row in facility_df.iterrows():
    try:
        fixed_cost[row['Facility']] = float(row['FixedCost'])
    except Exception:
        raise ValueError(f"Invalid FixedCost for facility {row['Facility']}")
shipping_costs_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP11/shipping_costs.csv'
shipping_df = pd.read_csv(shipping_costs_path, dtype=str, keep_default_na=False)
if 'Origin' not in shipping_df.columns:
    raise ValueError('shipping_costs.csv missing required Origin column')
distribution_centers = [col for col in shipping_df.columns if col != 'Origin']
cost = {}
for idx, row in shipping_df.iterrows():
    origin = row['Origin']
    if origin not in facilities:
        raise ValueError(f'Shipping cost origin {origin} not in facilities list')
    cost[origin] = {}
    for dest in distribution_centers:
        try:
            cost[origin][dest] = float(row[dest])
        except Exception:
            raise ValueError(f'Invalid shipping cost from {origin} to {dest}')
demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP11/demand_requirements.csv'
demand_df = pd.read_csv(demand_path, dtype=str, keep_default_na=False)
if 'Destination' not in demand_df.columns or 'Demand' not in demand_df.columns:
    raise ValueError('demand_requirements.csv missing required columns')
demand = {}
for idx, row in demand_df.iterrows():
    dest = row['Destination']
    if dest not in distribution_centers:
        raise ValueError(f'Demand destination {dest} not in distribution_centers list')
    try:
        demand[dest] = float(row['Demand'])
    except Exception:
        raise ValueError(f'Invalid demand for destination {dest}')
if set(facilities) != set(cost.keys()):
    raise ValueError('Mismatch between facilities and shipping cost origins')
if set(distribution_centers) != set(demand.keys()):
    raise ValueError('Mismatch between distribution centers and demand destinations')
for i in facilities:
    if set(cost[i].keys()) != set(distribution_centers):
        raise ValueError(f'Shipping cost for facility {i} missing some distribution centers')
M = sum((demand[j] for j in distribution_centers))
m = gp.Model('ElectroTech_UFLP')
x_vars = m.addVars(facilities, distribution_centers, lb=0, vtype=GRB.CONTINUOUS, name='')
y_vars = m.addVars(facilities, vtype=GRB.BINARY, name='')
m.setObjective(gp.quicksum((fixed_cost[i] * y_vars[i] for i in facilities)) + gp.quicksum((cost[i][j] * x_vars[i, j] for i in facilities for j in distribution_centers)), GRB.MINIMIZE)
m.addConstrs((gp.quicksum((x_vars[i, j] for i in facilities)) == demand[j] for j in distribution_centers), name='')
m.addConstrs((gp.quicksum((x_vars[i, j] for j in distribution_centers)) <= M * y_vars[i] for i in facilities), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')