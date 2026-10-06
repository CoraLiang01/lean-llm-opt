import gurobipy as gp
import pandas as pd
import numpy as np
import re
customer_demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP7/customer_demand.csv'
supply_capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP7/supply_capacity.csv'
transportation_costs_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP7/transportation_costs.csv'
df_demand = pd.read_csv(customer_demand_path, sep=',')
df_supply = pd.read_csv(supply_capacity_path, sep=',')
df_cost = pd.read_csv(transportation_costs_path, sep=',')
stores = df_demand['customer'].astype(str).tolist()
warehouses = df_supply['region'].astype(str).tolist()
demand = df_demand.set_index('customer')['demand'].astype(float).to_dict()
supply_capacity = df_supply.set_index('region')['supply_capacity'].astype(float).to_dict()
df_cost = df_cost.rename(columns={'Unnamed: 0': 'region'})
df_cost['region'] = df_cost['region'].astype(str)
cost = {}
for w in warehouses:
    row = df_cost[df_cost['region'] == w]
    if row.empty:
        raise ValueError(f"Warehouse '{w}' not found in transportation_costs.csv")
    for s in stores:
        if s not in row.columns:
            raise ValueError(f"Store '{s}' not found as column in transportation_costs.csv")
        cost[w, s] = float(row.iloc[0][s])
if set(stores) != set(df_cost.columns) - {'region'}:
    raise ValueError('Mismatch between stores in customer_demand.csv and transportation_costs.csv columns')
if set(warehouses) != set(df_cost['region']):
    raise ValueError('Mismatch between warehouses in supply_capacity.csv and transportation_costs.csv rows')
m = gp.Model('GreenMart_Transportation')
x = m.addVars(warehouses, stores, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((cost[w, s] * x[w, s] for w in warehouses for s in stores)), gp.GRB.MINIMIZE)
for s in stores:
    m.addConstr(gp.quicksum((x[w, s] for w in warehouses)) == demand[s], name=f'demand_{s}')
for w in warehouses:
    m.addConstr(gp.quicksum((x[w, s] for s in stores)) <= supply_capacity[w], name=f'supply_{w}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total transportation cost: {m.objVal:.2f}')
    print('--- Optimal Shipping Plan (quantities shipped from warehouse to store) ---')
    for w in warehouses:
        for s in stores:
            shipped = x[w, s].X
            if shipped > 1e-06:
                print(f'  Warehouse {w} -> Store {s}: {shipped:.2f} units')
else:
    print(f'No optimal solution found. Status: {m.status}')