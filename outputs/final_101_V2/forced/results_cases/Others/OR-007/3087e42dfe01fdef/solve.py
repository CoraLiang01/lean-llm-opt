import gurobipy as gp
import pandas as pd
import numpy as np
import re
customer_demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP7/customer_demand.csv'
supply_capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP7/supply_capacity.csv'
transportation_costs_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP7/transportation_costs.csv'
df_demand = pd.read_csv(customer_demand_path, sep=',')
df_demand['customer'] = df_demand['customer'].astype(str).str.strip()
customers = df_demand['customer'].tolist()
demand_dict = dict(zip(df_demand['customer'], df_demand['demand']))
df_supply = pd.read_csv(supply_capacity_path, sep=',')
df_supply['region'] = df_supply['region'].astype(str).str.strip()
warehouses = df_supply['region'].tolist()
supply_dict = dict(zip(df_supply['region'], df_supply['supply_capacity']))
df_costs = pd.read_csv(transportation_costs_path, sep=',')
df_costs['Unnamed: 0'] = df_costs['Unnamed: 0'].astype(str).str.strip()
cost_matrix = {}
for idx, row in df_costs.iterrows():
    w = row['Unnamed: 0']
    for s in customers:
        cost_matrix[w, s] = float(row[s])
for w in warehouses:
    if w not in df_costs['Unnamed: 0'].values:
        raise ValueError(f"Warehouse '{w}' from supply_capacity.csv not found in transportation_costs.csv")
for s in customers:
    if s not in df_costs.columns:
        raise ValueError(f"Customer '{s}' from customer_demand.csv not found in transportation_costs.csv columns")
m = gp.Model('GreenMart_Transportation')
x = m.addVars(warehouses, customers, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((cost_matrix[w, s] * x[w, s] for w in warehouses for s in customers)), gp.GRB.MINIMIZE)
for s in customers:
    m.addConstr(gp.quicksum((x[w, s] for w in warehouses)) == demand_dict[s], name=f'demand_{s}')
for w in warehouses:
    m.addConstr(gp.quicksum((x[w, s] for s in customers)) <= supply_dict[w], name=f'supply_{w}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total transportation cost: {m.objVal:.2f}')
    print('--- Optimal Shipping Plan (quantities shipped from each warehouse to each store) ---')
    for w in warehouses:
        for s in customers:
            shipped = x[w, s].X
            if shipped > 1e-06:
                print(f'  Warehouse {w} -> Store {s}: {shipped:.2f} units')
else:
    print(f'No optimal solution found. Status: {m.status}')