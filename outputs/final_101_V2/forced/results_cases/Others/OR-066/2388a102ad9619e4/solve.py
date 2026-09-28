import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP10/demand.csv'
fixed_cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP10/fixed_cost.csv'
transport_cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP10/transportation_costs.csv'
df_demand = pd.read_csv(demand_path, sep=',')
customers = df_demand['customer'].astype(str).str.strip().tolist()
demand_dict = dict(zip(df_demand['customer'].astype(str).str.strip(), df_demand['demand']))
df_fixed = pd.read_csv(fixed_cost_path, sep=',')
suppliers = df_fixed['Unnamed: 0'].astype(str).str.strip().tolist()
fixed_cost_dict = dict(zip(df_fixed['Unnamed: 0'].astype(str).str.strip(), df_fixed['fixed_costs']))
df_trans = pd.read_csv(transport_cost_path, sep=',')
df_trans['Unnamed: 0'] = df_trans['Unnamed: 0'].astype(str).str.strip()
trans_suppliers = df_trans['Unnamed: 0'].tolist()
trans_customers = [col for col in df_trans.columns if col != 'Unnamed: 0']
if set(suppliers) != set(trans_suppliers):
    raise ValueError(f'Supplier IDs in fixed_cost.csv and transportation_costs.csv do not match: {suppliers} vs {trans_suppliers}')
if set(customers) != set(trans_customers):
    raise ValueError(f'Customer IDs in demand.csv and transportation_costs.csv do not match: {customers} vs {trans_customers}')
transport_cost = {}
for idx, row in df_trans.iterrows():
    i = row['Unnamed: 0']
    for j in customers:
        transport_cost[i, j] = float(row[j])
m = gp.Model('UFLP')
y = m.addVars(suppliers, vtype=gp.GRB.BINARY, name='')
x = m.addVars(suppliers, customers, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
fixed_cost_expr = gp.quicksum((fixed_cost_dict[i] * y[i] for i in suppliers))
transport_cost_expr = gp.quicksum((transport_cost[i, j] * x[i, j] for i in suppliers for j in customers))
m.setObjective(fixed_cost_expr + transport_cost_expr, gp.GRB.MINIMIZE)
for j in customers:
    m.addConstr(gp.quicksum((x[i, j] for i in suppliers)) == demand_dict[j], name=f'demand_{j}')
M = sum(demand_dict.values())
for i in suppliers:
    for j in customers:
        m.addConstr(x[i, j] <= M * y[i], name=f'link_{i}_{j}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total cost: {m.objVal:.2f}\n')
    print('--- Supplier Activation ---')
    for i in suppliers:
        print(f"  Supplier {i}: {('Activated' if y[i].X > 0.5 else 'Not Activated')} (y={int(round(y[i].X))})")
    print('\n--- Shipment Plan (x[i,j]) ---')
    for i in suppliers:
        for j in customers:
            if x[i, j].X > 1e-06:
                print(f'  Supplier {i} -> Customer {j}: {x[i, j].X:.2f}')
else:
    print(f'No optimal solution found. Status: {m.status}')