import gurobipy as gp
import pandas as pd
import numpy as np
import re
customer_demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP2/customer_demand.csv'
supply_capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP2/supply_capacity.csv'
transportation_costs_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP2/transportation_costs.csv'
df_demand = pd.read_csv(customer_demand_path, sep=',')
df_demand['customer'] = df_demand['customer'].astype(str).str.strip()
customers = df_demand['customer'].tolist()
customer_set = set(customers)
customer_demand = dict(zip(df_demand['customer'], df_demand['demand']))
df_supply = pd.read_csv(supply_capacity_path, sep=',')
df_supply['store'] = df_supply['Unnamed: 0'].astype(str).str.strip()
stores = df_supply['store'].tolist()
store_set = set(stores)
supply_capacity = dict(zip(df_supply['store'], df_supply['supply_capacity']))
df_cost = pd.read_csv(transportation_costs_path, sep=',')
df_cost['store'] = df_cost['Unnamed: 0'].astype(str).str.strip()
cost_customer_cols = [col for col in df_cost.columns if col in customer_set]
if set(cost_customer_cols) != customer_set:
    missing = customer_set - set(cost_customer_cols)
    raise ValueError(f'Missing transportation cost columns for customers: {missing}')
if set(df_cost['store']) != store_set:
    missing = store_set - set(df_cost['store'])
    extra = set(df_cost['store']) - store_set
    if missing:
        raise ValueError(f'Missing transportation cost rows for stores: {missing}')
    if extra:
        raise ValueError(f'Extra stores in transportation_costs.csv not in supply_capacity.csv: {extra}')
cost = {}
for _, row in df_cost.iterrows():
    s = row['store']
    for c in customers:
        cost[s, c] = float(row[c])
if set(customer_demand.keys()) != customer_set:
    raise ValueError('Mismatch in customer demand keys and customer set.')
if set(supply_capacity.keys()) != store_set:
    raise ValueError('Mismatch in supply capacity keys and store set.')
for s in stores:
    for c in customers:
        if (s, c) not in cost:
            raise ValueError(f'Missing transportation cost for store {s}, customer {c}')
m = gp.Model('Walmart_Transportation')
x = m.addVars(stores, customers, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((cost[s, c] * x[s, c] for s in stores for c in customers)), gp.GRB.MINIMIZE)
for c in customers:
    m.addConstr(gp.quicksum((x[s, c] for s in stores)) == customer_demand[c], name=f'demand_{c}')
for s in stores:
    m.addConstr(gp.quicksum((x[s, c] for c in customers)) <= supply_capacity[s], name=f'supply_{s}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total transportation cost: {m.objVal:.6f}')
    print('Optimal transportation plan (quantities shipped from each store to each customer):')
    for s in stores:
        for c in customers:
            val = x[s, c].X
            if val > 1e-06:
                print(f'  Store {s} -> Customer {c}: {val:.6f}')
else:
    print(f'No optimal solution found. Status: {m.status}')