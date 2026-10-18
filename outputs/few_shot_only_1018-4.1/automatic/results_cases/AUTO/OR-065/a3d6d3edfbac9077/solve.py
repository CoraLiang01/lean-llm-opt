import pandas as pd
import gurobipy as gp
from gurobipy import GRB
demand_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP9/demand.csv', dtype=str, keep_default_na=False)
customers = demand_df['customer'].tolist()
demand = {}
for idx, row in demand_df.iterrows():
    cust = row['customer']
    val = float(row['demand'])
    demand[cust] = val
fixed_cost_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP9/fixed_cost.csv', dtype=str, keep_default_na=False)
warehouses = fixed_cost_df['Unnamed: 0'].tolist()
fixed_cost = {}
for idx, row in fixed_cost_df.iterrows():
    wh = row['Unnamed: 0']
    val = float(row['fixed_costs'])
    fixed_cost[wh] = val
trans_costs_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP9/transportation_costs.csv', dtype=str, keep_default_na=False)
cost = {}
for idx, row in trans_costs_df.iterrows():
    wh = row['Unnamed: 0']
    cost[wh] = {}
    for cust in customers:
        val = float(row[cust])
        cost[wh][cust] = val
if set(warehouses) != set(cost.keys()):
    raise ValueError('Mismatch between warehouses in fixed_cost and transportation_costs')
if set(customers) != set(demand.keys()):
    raise ValueError('Mismatch between customers in demand and transportation_costs')
M = sum((demand[cust] for cust in customers))
m = gp.Model('Bandcamp_UFLP')
x_vars = m.addVars(warehouses, customers, lb=0, vtype=GRB.CONTINUOUS, name='')
y_vars = m.addVars(warehouses, vtype=GRB.BINARY, name='')
m.setObjective(gp.quicksum((cost[i][j] * x_vars[i, j] for i in warehouses for j in customers)) + gp.quicksum((fixed_cost[i] * y_vars[i] for i in warehouses)), GRB.MINIMIZE)
m.addConstrs((gp.quicksum((x_vars[i, j] for i in warehouses)) == demand[j] for j in customers), name='')
m.addConstrs((gp.quicksum((x_vars[i, j] for j in customers)) <= M * y_vars[i] for i in warehouses), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')