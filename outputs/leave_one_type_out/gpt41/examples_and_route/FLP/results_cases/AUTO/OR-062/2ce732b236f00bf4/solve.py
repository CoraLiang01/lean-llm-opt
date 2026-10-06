import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
demand_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP6/demand.csv', sep=',')
fixed_cost_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP6/fixed_cost.csv', sep=',')
trans_cost_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP6/transportation_costs.csv', sep=',')
customers = demand_df['Customer'].astype(str).tolist()

def normalize_name(s):
    return re.sub('\\s+', ' ', str(s)).strip().upper()
fixed_cost_df['supplier_norm'] = fixed_cost_df['Unnamed: 0'].apply(normalize_name)
trans_cost_df['supplier_norm'] = trans_cost_df['Unnamed: 0'].apply(normalize_name)
supplier_norm_to_orig = {row['supplier_norm']: row['Unnamed: 0'] for _, row in fixed_cost_df.iterrows()}
suppliers = [row['Unnamed: 0'] for _, row in fixed_cost_df.iterrows()]
supplier_norms = [normalize_name(s) for s in suppliers]
store_cols = [col for col in trans_cost_df.columns if col not in ['Unnamed: 0', 'supplier_norm']]
if len(customers) != len(store_cols):
    raise ValueError('Number of customers in demand.csv does not match number of stores in transportation_costs.csv columns.')
customer_to_storecol = dict(zip(customers, store_cols))
storecol_to_customer = {v: k for k, v in customer_to_storecol.items()}
fixed_costs = {}
for _, row in fixed_cost_df.iterrows():
    supplier = row['Unnamed: 0']
    supplier_norm = normalize_name(supplier)
    fixed_costs[supplier] = float(row['fixed_costs'])
demand = {}
for _, row in demand_df.iterrows():
    customer = str(row['Customer'])
    demand[customer] = int(row['demand'])
trans_cost = {}
for idx, row in trans_cost_df.iterrows():
    supplier_norm = row['supplier_norm']
    if supplier_norm not in supplier_norm_to_orig:
        raise ValueError(f"Supplier '{row['Unnamed: 0']}' in transportation_costs.csv not found in fixed_cost.csv")
    supplier = supplier_norm_to_orig[supplier_norm]
    for store_col in store_cols:
        customer = storecol_to_customer[store_col]
        cost = float(row[store_col])
        trans_cost[supplier, customer] = cost
I = suppliers
J = customers
M = sum((demand[j] for j in J))
m = gp.Model('UFLP_Iowa_Liquor')
y = m.addVars(I, vtype=gp.GRB.BINARY, name='')
x = m.addVars(I, J, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
total_fixed = gp.quicksum((fixed_costs[i] * y[i] for i in I))
total_trans = gp.quicksum((trans_cost[i, j] * x[i, j] for i in I for j in J))
m.setObjective(total_fixed + total_trans, gp.GRB.MINIMIZE)
for j in J:
    m.addConstr(gp.quicksum((x[i, j] for i in I)) == demand[j], name=f'demand_{j}')
for i in I:
    for j in J:
        m.addConstr(x[i, j] <= M * y[i], name=f'link_{i}_{j}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f}')
    print('\n--- Supplier Activation ---')
    for i in I:
        print(f"  Supplier '{i}': {('OPEN' if y[i].X > 0.5 else 'CLOSED')} (y={int(round(y[i].X))})")
    print('\n--- Shipment Plan (nonzero only) ---')
    for i in I:
        for j in J:
            if x[i, j].X > 1e-06:
                print(f"  {x[i, j].X:.2f} units from Supplier '{i}' to Customer '{j}' (cost per unit: {trans_cost[i, j]:.2f})")
else:
    print(f'No optimal solution found. Status: {m.status}')