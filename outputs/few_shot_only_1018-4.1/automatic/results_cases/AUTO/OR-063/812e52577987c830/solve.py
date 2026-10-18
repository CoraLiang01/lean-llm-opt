import pandas as pd
import gurobipy as gp
from gurobipy import GRB
demand_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP7/demand.csv', dtype=str, keep_default_na=False)
demand = {}
customers = []
for idx, row in demand_df.iterrows():
    cust = row[demand_df.columns[0]].strip()
    val = float(row[demand_df.columns[1]])
    demand[cust] = val
    customers.append(cust)
fixed_cost_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP7/fixed_cost.csv', dtype=str, keep_default_na=False)
fixed_cost = {}
warehouses = []
for idx, row in fixed_cost_df.iterrows():
    wh = row[fixed_cost_df.columns[0]].strip()
    val = float(row[fixed_cost_df.columns[1]])
    fixed_cost[wh] = val
    warehouses.append(wh)
trans_costs_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP7/transportation_costs.csv', dtype=str, keep_default_na=False)
cost = {}
for idx, row in trans_costs_df.iterrows():
    wh = row[trans_costs_df.columns[0]].strip()
    cost[wh] = {}
    for cust in customers:
        val = float(row[cust])
        cost[wh][cust] = val
if set(demand.keys()) != set(customers):
    raise ValueError('Mismatch in customer identifiers between demand and cost data.')
if set(fixed_cost.keys()) != set(warehouses):
    raise ValueError('Mismatch in warehouse identifiers between fixed cost and cost data.')
for wh in warehouses:
    if set(cost[wh].keys()) != set(customers):
        raise ValueError(f'Mismatch in customer identifiers for warehouse {wh} in cost data.')
M = sum(demand.values())
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