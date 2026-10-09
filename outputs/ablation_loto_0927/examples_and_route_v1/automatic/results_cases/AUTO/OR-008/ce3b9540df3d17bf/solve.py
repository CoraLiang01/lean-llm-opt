import gurobipy as gp
import pandas as pd
import numpy as np
import re
customer_demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP8/customer_demand.csv'
supply_capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP8/supply_capacity.csv'
transportation_costs_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP8/transportation_costs.csv'
df_demand = pd.read_csv(customer_demand_path, sep=',')
df_supply = pd.read_csv(supply_capacity_path, sep=',')
df_costs = pd.read_csv(transportation_costs_path, sep=',')
customers = df_demand['Customers'].astype(str).tolist()
demand = dict(zip(df_demand['Customers'].astype(str), df_demand['demand']))
suppliers = df_supply['Suppliers'].astype(str).tolist()
supply_capacity = dict(zip(df_supply['Suppliers'].astype(str), df_supply['supply_capacity']))
df_costs = df_costs.rename(columns={'Unnamed: 0': 'Suppliers'})
df_costs['Suppliers'] = df_costs['Suppliers'].astype(str).str.strip()
cost_matrix = df_costs.set_index('Suppliers')
missing_suppliers = set(suppliers) - set(cost_matrix.index)
missing_customers = set(customers) - set(cost_matrix.columns)
if missing_suppliers:
    raise ValueError(f'Missing suppliers in transportation_costs.csv: {missing_suppliers}')
if missing_customers:
    raise ValueError(f'Missing customers in transportation_costs.csv: {missing_customers}')
cost = {}
for s in suppliers:
    for c in customers:
        cost[s, c] = float(cost_matrix.loc[s, c])
m = gp.Model('FreshMart_Transportation')
x = m.addVars(suppliers, customers, vtype=gp.GRB.CONTINUOUS, lb=0.0, name='')
m.setObjective(gp.quicksum((cost[s, c] * x[s, c] for s in suppliers for c in customers)), sense=gp.GRB.MINIMIZE)
for s in suppliers:
    m.addConstr(gp.quicksum((x[s, c] for c in customers)) <= supply_capacity[s], name=f'supply_{s}')
for c in customers:
    m.addConstr(gp.quicksum((x[s, c] for s in suppliers)) == demand[c], name=f'demand_{c}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total transportation cost: {m.objVal:.2f}')
    print('\n--- Optimal Shipment Plan (amounts shipped from each supplier to each customer) ---')
    for s in suppliers:
        for c in customers:
            shipped = x[s, c].X
            if shipped > 1e-06:
                print(f'  {s} → {c}: {shipped:.2f}')
else:
    print(f'No optimal solution found. Status: {m.status}')