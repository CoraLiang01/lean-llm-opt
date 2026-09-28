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
suppliers = df_supply['Suppliers'].astype(str).tolist()
demand = df_demand.set_index('Customers')['demand'].astype(float).to_dict()
supply_capacity = df_supply.set_index('Suppliers')['supply_capacity'].astype(float).to_dict()
df_costs = df_costs.rename(columns={'Unnamed: 0': 'Suppliers'})
df_costs['Suppliers'] = df_costs['Suppliers'].astype(str).str.strip()
cost = {}
for s in suppliers:
    row = df_costs[df_costs['Suppliers'].str.strip() == s]
    if row.empty:
        raise ValueError(f"Supplier '{s}' not found in transportation_costs.csv")
    row = row.iloc[0]
    cost[s] = {}
    for c in customers:
        if c not in row:
            raise ValueError(f"Customer '{c}' not found in transportation_costs.csv columns")
        cost[s][c] = float(row[c])
if set(demand.keys()) != set(customers):
    raise ValueError('Mismatch between customer_demand.csv and customers index set')
if set(supply_capacity.keys()) != set(suppliers):
    raise ValueError('Mismatch between supply_capacity.csv and suppliers index set')
for s in suppliers:
    if s not in cost:
        raise ValueError(f"Missing cost row for supplier '{s}'")
    if set(cost[s].keys()) != set(customers):
        raise ValueError(f"Cost row for supplier '{s}' missing some customers")
m = gp.Model('FreshMart_Transportation')
x = m.addVars(suppliers, customers, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((cost[s][c] * x[s, c] for s in suppliers for c in customers)), sense=gp.GRB.MINIMIZE)
for c in customers:
    m.addConstr(gp.quicksum((x[s, c] for s in suppliers)) == demand[c], name=f'demand_{c}')
for s in suppliers:
    m.addConstr(gp.quicksum((x[s, c] for c in customers)) <= supply_capacity[s], name=f'supply_{s}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total transportation cost: {m.objVal:.2f}')
    print('\n--- Optimal Shipment Plan (units shipped from each supplier to each customer) ---')
    for s in suppliers:
        for c in customers:
            shipped = x[s, c].X
            if shipped > 1e-06:
                print(f'  {s} -> {c}: {shipped:.2f}')
else:
    print(f'No optimal solution found. Status: {m.status}')