import gurobipy as gp
import pandas as pd
import numpy as np
import re
customer_demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP7/customer_demand.csv'
supply_capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP7/supply_capacity.csv'
transportation_costs_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP7/transportation_costs.csv'
df_demand = pd.read_csv(customer_demand_path, sep=',')
df_demand['customer'] = df_demand['customer'].astype(str).str.strip()
stores = df_demand['customer'].tolist()
demand_dict = dict(zip(df_demand['customer'], df_demand['demand']))
df_supply = pd.read_csv(supply_capacity_path, sep=',')
df_supply['region'] = df_supply['region'].astype(str).str.strip()
warehouses = df_supply['region'].tolist()
supply_dict = dict(zip(df_supply['region'], df_supply['supply_capacity']))
df_costs = pd.read_csv(transportation_costs_path, sep=',')
df_costs['Unnamed: 0'] = df_costs['Unnamed: 0'].astype(str).str.strip()
cost_warehouses = df_costs['Unnamed: 0'].tolist()
cost_stores = [col for col in df_costs.columns if col != 'Unnamed: 0']
if set(warehouses) != set(cost_warehouses):
    raise ValueError(f'Mismatch between warehouses in supply_capacity.csv and transportation_costs.csv: {set(warehouses)} vs {set(cost_warehouses)}')
if set(stores) != set(cost_stores):
    raise ValueError(f'Mismatch between stores in customer_demand.csv and transportation_costs.csv: {set(stores)} vs {set(cost_stores)}')
cost = {}
for (_, row) in df_costs.iterrows():
    w = row['Unnamed: 0']
    for d in stores:
        cost[w, d] = float(row[d])
m = gp.Model('GreenMart_Transportation')
x = m.addVars(warehouses, stores, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((cost[w, d] * x[w, d] for w in warehouses for d in stores)), gp.GRB.MINIMIZE)
for d in stores:
    m.addConstr(gp.quicksum((x[w, d] for w in warehouses)) == demand_dict[d], name=f'demand_{d}')
for w in warehouses:
    m.addConstr(gp.quicksum((x[w, d] for d in stores)) <= supply_dict[w], name=f'supply_{w}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total transportation cost: {m.objVal:.2f}')
    print('--- Optimal Shipment Plan (quantities shipped from each warehouse to each store) ---')
    for w in warehouses:
        for d in stores:
            shipped = x[w, d].X
            if shipped > 1e-06:
                print(f'  Warehouse {w} -> Store {d}: {shipped:.2f} units')
else:
    print(f'No optimal solution found. Status: {m.status}')