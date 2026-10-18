import pandas as pd
import gurobipy as gp
from gurobipy import GRB
demand_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP1/demand.csv', dtype=str, keep_default_na=False)
demand = {}
for idx, row in demand_df.iterrows():
    cust = row[demand_df.columns[0]].strip()
    val = float(row[demand_df.columns[1]])
    demand[cust] = val
fixed_cost_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP1/fixed_cost.csv', dtype=str, keep_default_na=False)
fixed_cost = {}
for idx, row in fixed_cost_df.iterrows():
    fac = row[fixed_cost_df.columns[0]].strip()
    val = float(row[fixed_cost_df.columns[1]])
    fixed_cost[fac] = val
trans_cost_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP1/transportation_costs.csv', dtype=str, keep_default_na=False)
suppliers = [row[trans_cost_df.columns[0]].strip() for idx, row in trans_cost_df.iterrows()]
customers = [col.strip() for col in trans_cost_df.columns[1:]]
cost = {}
for idx, row in trans_cost_df.iterrows():
    fac = row[trans_cost_df.columns[0]].strip()
    cost[fac] = {}
    for cust in customers:
        val = float(row[cust])
        cost[fac][cust] = val
if set(demand.keys()) != set(customers):
    raise ValueError('Mismatch between demand customers and transportation cost columns')
if set(fixed_cost.keys()) != set(suppliers):
    raise ValueError('Mismatch between fixed cost suppliers and transportation cost rows')
M = sum(demand.values())
m = gp.Model('UFLP_OriginalQuery')
x_vars = m.addVars(suppliers, customers, lb=0, vtype=GRB.CONTINUOUS, name='')
y_vars = m.addVars(suppliers, vtype=GRB.BINARY, name='')
m.setObjective(gp.quicksum((cost[i][j] * x_vars[i, j] for i in suppliers for j in customers)) + gp.quicksum((fixed_cost[i] * y_vars[i] for i in suppliers)), GRB.MINIMIZE)
m.addConstrs((gp.quicksum((x_vars[i, j] for i in suppliers)) == demand[j] for j in customers), name='')
m.addConstrs((gp.quicksum((x_vars[i, j] for j in customers)) <= M * y_vars[i] for i in suppliers), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')