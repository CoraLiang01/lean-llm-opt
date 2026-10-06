import gurobipy as gp
import pandas as pd
import numpy as np
import math
demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP5/demand.csv'
fixed_cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP5/fixed_cost.csv'
transport_cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP5/transportation_costs.csv'
demand_df = pd.read_csv(demand_path, sep=',')
fixed_cost_df = pd.read_csv(fixed_cost_path, sep=',')
transport_df = pd.read_csv(transport_cost_path, sep=',')
suppliers = fixed_cost_df['Unnamed: 0'].astype(str).str.strip().tolist()
customers = demand_df['customer'].astype(str).str.strip().tolist()
transport_suppliers = transport_df['Unnamed: 0'].astype(str).str.strip().tolist()
transport_customers = [c for c in transport_df.columns if c != 'Unnamed: 0']
if set(suppliers) != set(transport_suppliers):
    raise ValueError(f'Supplier mismatch between fixed_cost.csv and transportation_costs.csv: {set(suppliers)} vs {set(transport_suppliers)}')
if set(customers) != set(transport_customers):
    raise ValueError(f'Customer mismatch between demand.csv and transportation_costs.csv: {set(customers)} vs {set(transport_customers)}')
fixed_costs = dict(zip(fixed_cost_df['Unnamed: 0'].astype(str).str.strip(), fixed_cost_df['fixed_costs']))
demand = dict(zip(demand_df['customer'].astype(str).str.strip(), demand_df['demand']))
transport_costs = {}
for (idx, row) in transport_df.iterrows():
    s = str(row['Unnamed: 0']).strip()
    for c in customers:
        transport_costs[s, c] = float(row[c])
I = suppliers
J = customers
M = sum((demand[j] for j in J))
m = gp.Model('UFLP_Superstore')
x = m.addVars(I, J, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
y = m.addVars(I, vtype=gp.GRB.BINARY, name='')
obj = gp.quicksum((fixed_costs[i] * y[i] for i in I)) + gp.quicksum((transport_costs[i, j] * x[i, j] for i in I for j in J))
m.setObjective(obj, gp.GRB.MINIMIZE)
for j in J:
    m.addConstr(gp.quicksum((x[i, j] for i in I)) == demand[j], name=f'demand_{j}')
for i in I:
    for j in J:
        m.addConstr(x[i, j] <= M * y[i], name=f'link_{i}_{j}')
m.Params.MIPGap = 0.0001
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for i in I:
        print(f'y[{i}] {y[i].VarName} {y[i].X}')
    for i in I:
        for j in J:
            print(f'x[{i},{j}] {x[i, j].VarName} {x[i, j].X}')
else:
    print(f'Solver status: {m.status}')