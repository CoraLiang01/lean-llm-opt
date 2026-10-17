import gurobipy as gp
from gurobipy import GRB
import pandas as pd
demand_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP2/demand.csv', dtype=str, keep_default_na=False)
fixed_cost_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP2/fixed_cost.csv', dtype=str, keep_default_na=False)
transport_cost_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP2/transportation_costs.csv', dtype=str, keep_default_na=False)
supplier_ids = list(fixed_cost_df['supplier'])
customer_ids = list(demand_df['dealership'])
demand = {}
for _, row in demand_df.iterrows():
    demand[row['dealership']] = float(row['demand'])
fixed_cost = {}
for _, row in fixed_cost_df.iterrows():
    fixed_cost[row['supplier']] = float(row['fixed_cost'])
cost = {}
for _, row in transport_cost_df.iterrows():
    supplier = row['supplier']
    cost[supplier] = {}
    for cust in customer_ids:
        cost_val = row[cust]
        if cost_val == '':
            raise ValueError(f'Missing transportation cost for supplier {supplier}, customer {cust}')
        cost[supplier][cust] = float(cost_val)
if set(supplier_ids) != set(cost.keys()):
    raise ValueError('Mismatch between supplier IDs in fixed_cost and transportation_costs')
for s in supplier_ids:
    if set(customer_ids) != set(cost[s].keys()):
        raise ValueError(f'Mismatch between customer IDs in transportation_costs for supplier {s}')
M = sum((demand[cust] for cust in customer_ids))
m = gp.Model('Colorado_Motor_Vehicle_UFLP')
x_vars = m.addVars(supplier_ids, customer_ids, lb=0, vtype=GRB.CONTINUOUS, name='')
y_vars = m.addVars(supplier_ids, vtype=GRB.BINARY, name='')
m.setObjective(gp.quicksum((cost[s][c] * x_vars[s, c] for s in supplier_ids for c in customer_ids)) + gp.quicksum((fixed_cost[s] * y_vars[s] for s in supplier_ids)), GRB.MINIMIZE)
m.addConstrs((gp.quicksum((x_vars[s, c] for s in supplier_ids)) == demand[c] for c in customer_ids), name='')
m.addConstrs((gp.quicksum((x_vars[s, c] for c in customer_ids)) <= M * y_vars[s] for s in supplier_ids), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')