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
cost_warehouses = df_costs['Unnamed: 0'].tolist()
cost_customers = [col for col in df_costs.columns if col != 'Unnamed: 0']
missing_warehouses = set(warehouses) - set(cost_warehouses)
missing_customers = set(customers) - set(cost_customers)
if missing_warehouses:
    raise ValueError(f'Missing warehouses in transportation_costs.csv: {missing_warehouses}')
if missing_customers:
    raise ValueError(f'Missing customers in transportation_costs.csv: {missing_customers}')
cost = {}
for _, row in df_costs.iterrows():
    w = row['Unnamed: 0']
    for s in customers:
        cost[w, s] = float(row[s])
m = gp.Model('GreenMart_Transportation')
x = m.addVars(warehouses, customers, vtype=gp.GRB.CONTINUOUS, lb=0.0, name='')
m.setObjective(gp.quicksum((cost[w, s] * x[w, s] for w in warehouses for s in customers)), gp.GRB.MINIMIZE)
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