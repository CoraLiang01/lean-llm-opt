import pandas as pd
import gurobipy as gp
from gurobipy import GRB
cost_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP12/cost.csv', dtype=str, keep_default_na=False)
plants = cost_df['plant'].tolist()
fixed_cost = {}
capacity = {}
cost = {}
customer_cols = [col for col in cost_df.columns if col.startswith('C')]
for idx, row in cost_df.iterrows():
    plant = row['plant']
    fixed_cost[plant] = float(row['fixed_cost'])
    capacity[plant] = float(row['capacity'])
    cost[plant] = {}
    for cust in customer_cols:
        cost[plant][cust] = float(row[cust])
demand_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP12/demand.csv', dtype=str, keep_default_na=False)
customers = demand_df['customer'].tolist()
demand = {}
for idx, row in demand_df.iterrows():
    cust = row['customer']
    demand[cust] = float(row['demand'])
if set(cost[plants[0]].keys()) != set(customers):
    raise ValueError('Mismatch between cost.csv customer columns and demand.csv customers.')
if set(fixed_cost.keys()) != set(plants):
    raise ValueError('Mismatch between plant identifiers in cost.csv.')
m = gp.Model('UFLP12')
x_vars = m.addVars(plants, customers, lb=0, vtype=GRB.CONTINUOUS, name='')
y_vars = m.addVars(plants, vtype=GRB.BINARY, name='')
m.setObjective(gp.quicksum((fixed_cost[i] * y_vars[i] for i in plants)) + gp.quicksum((cost[i][j] * x_vars[i, j] for i in plants for j in customers)), GRB.MINIMIZE)
m.addConstrs((gp.quicksum((x_vars[i, j] for i in plants)) == demand[j] for j in customers), name='')
m.addConstrs((gp.quicksum((x_vars[i, j] for j in customers)) <= capacity[i] * y_vars[i] for i in plants), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')