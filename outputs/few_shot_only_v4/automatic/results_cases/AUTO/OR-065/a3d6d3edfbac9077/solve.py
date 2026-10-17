import pandas as pd
import gurobipy as gp
from gurobipy import GRB
demand_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP9/demand.csv', dtype=str, keep_default_na=False)
fixed_cost_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP9/fixed_cost.csv', dtype=str, keep_default_na=False)
transport_costs_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP9/transportation_costs.csv', dtype=str, keep_default_na=False)
warehouses = list(fixed_cost_df['facility'])
musicians = list(demand_df['customer'])
demand = {}
for _, row in demand_df.iterrows():
    cust = row['customer']
    val = float(row['demand'])
    demand[cust] = val
fixed_cost = {}
for _, row in fixed_cost_df.iterrows():
    fac = row['facility']
    val = float(row['fixed_cost'])
    fixed_cost[fac] = val
cost = {fac: {} for fac in warehouses}
for _, row in transport_costs_df.iterrows():
    fac = row['facility']
    for cust in musicians:
        val = float(row[cust])
        cost[fac][cust] = val
if set(cost.keys()) != set(warehouses):
    raise ValueError('Mismatch between warehouse keys in cost and fixed_cost.')
for fac in warehouses:
    if set(cost[fac].keys()) != set(musicians):
        raise ValueError(f'Mismatch between customer keys in cost[{fac}] and demand.')
M = sum((demand[cust] for cust in musicians))
m = gp.Model('Bandcamp_Warehouse_Selection')
x_vars = m.addVars(warehouses, musicians, lb=0, vtype=GRB.CONTINUOUS, name='')
y_vars = m.addVars(warehouses, vtype=GRB.BINARY, name='')
m.setObjective(gp.quicksum((cost[fac][cust] * x_vars[fac, cust] for fac in warehouses for cust in musicians)) + gp.quicksum((fixed_cost[fac] * y_vars[fac] for fac in warehouses)), GRB.MINIMIZE)
m.addConstrs((gp.quicksum((x_vars[fac, cust] for fac in warehouses)) == demand[cust] for cust in musicians), name='')
m.addConstrs((gp.quicksum((x_vars[fac, cust] for cust in musicians)) <= M * y_vars[fac] for fac in warehouses), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')