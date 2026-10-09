import gurobipy as gp
import pandas as pd
import numpy as np
import re
customer_demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP3/customer_demand.csv'
supply_capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP3/supply_capacity.csv'
transportation_costs_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP3/transportation_costs.csv'
df_cust = pd.read_csv(customer_demand_path, sep=',')
df_cust['customer'] = df_cust['customer'].astype(str).str.strip()
customers = df_cust['customer'].tolist()
demand = dict(zip(df_cust['customer'], df_cust['demand']))
df_supp = pd.read_csv(supply_capacity_path, sep=',')
df_supp['supplier'] = df_supp['Unnamed: 0'].astype(str).str.strip()
suppliers = df_supp['supplier'].tolist()
supply_capacity = dict(zip(df_supp['supplier'], df_supp['supply_capacity']))
df_cost = pd.read_csv(transportation_costs_path, sep=',')
df_cost['supplier'] = df_cost['Unnamed: 0'].astype(str).str.strip()
missing_customers = [c for c in customers if c not in df_cost.columns]
if missing_customers:
    raise KeyError(f'Missing transportation cost columns for customers: {missing_customers}')
cost = {}
for (idx, row) in df_cost.iterrows():
    s = str(row['supplier']).strip()
    for c in customers:
        cost[s, c] = float(row[c])
cost_suppliers = set(df_cost['supplier'])
if set(suppliers) != cost_suppliers:
    raise ValueError(f'Supplier mismatch between supply_capacity and transportation_costs: {set(suppliers) ^ cost_suppliers}')
cost_customers = set(df_cost.columns) - {'Unnamed: 0', 'supplier'}
if set(customers) != cost_customers:
    raise ValueError(f'Customer mismatch between customer_demand and transportation_costs: {set(customers) ^ cost_customers}')
m = gp.Model('Transportation')
x = m.addVars(suppliers, customers, lb=0.0, name='')
m.setObjective(gp.quicksum((cost[s, c] * x[s, c] for s in suppliers for c in customers)), gp.GRB.MINIMIZE)
for s in suppliers:
    m.addConstr(gp.quicksum((x[s, c] for c in customers)) <= supply_capacity[s], name=f'supply_{s}')
for c in customers:
    m.addConstr(gp.quicksum((x[s, c] for s in suppliers)) == demand[c], name=f'demand_{c}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total transportation cost: {m.objVal:.2f}')
    print('\n--- Optimal Transportation Plan (units shipped) ---')
    for s in suppliers:
        for c in customers:
            shipped = x[s, c].X
            if shipped > 1e-06:
                print(f'  Supplier {s} -> Customer {c}: {shipped:.2f}')
else:
    print(f'No optimal solution found. Status: {m.status}')