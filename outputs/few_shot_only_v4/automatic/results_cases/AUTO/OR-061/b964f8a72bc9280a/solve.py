import gurobipy as gp
from gurobipy import GRB
import pandas as pd
demand_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP5/demand.csv', dtype=str, keep_default_na=False)
fixed_cost_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP5/fixed_cost.csv', dtype=str, keep_default_na=False)
transport_costs_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP5/transportation_costs.csv', dtype=str, keep_default_na=False)
suppliers = list(fixed_cost_df.iloc[:, 0])
customers = list(demand_df.iloc[:, 0])
demand = {}
for idx, row in demand_df.iterrows():
    cust = row.iloc[0]
    val = float(row.iloc[1])
    demand[cust] = val
fixed_cost = {}
for idx, row in fixed_cost_df.iterrows():
    sup = row.iloc[0]
    val = float(row.iloc[1])
    fixed_cost[sup] = val
cost = {}
header = list(transport_costs_df.columns)
customer_cols = header[1:]
if set(customer_cols) != set(customers):
    raise ValueError('Mismatch between customer columns in transportation_costs.csv and demand.csv')
for idx, row in transport_costs_df.iterrows():
    sup = row.iloc[0]
    cost[sup] = {}
    for j, cust in enumerate(customer_cols):
        cost[sup][cust] = float(row.iloc[j + 1])
if set(suppliers) != set(cost.keys()):
    raise ValueError('Mismatch between suppliers in fixed_cost.csv and transportation_costs.csv')
for sup in suppliers:
    if set(cost[sup].keys()) != set(customers):
        raise ValueError(f'Mismatch in customer keys for supplier {sup} in transportation_costs.csv')
M = sum((demand[cust] for cust in customers))
m = gp.Model('Superstore_UFLP')
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