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
demand = dict(zip(df_demand['customer'], df_demand['demand']))
df_supply = pd.read_csv(supply_capacity_path, sep=',')
df_supply['store'] = df_supply['Unnamed: 0'].astype(str).str.strip()
stores = df_supply['store'].tolist()
supply_capacity = dict(zip(df_supply['store'], df_supply['supply_capacity']))
df_cost = pd.read_csv(transportation_costs_path, sep=',')
df_cost['store'] = df_cost['Unnamed: 0'].astype(str).str.strip()
cost_columns = [c for c in df_cost.columns if re.fullmatch('C\\d+', c)]
missing_customers = set(customers) - set(cost_columns)
if missing_customers:
    raise ValueError(f'Missing cost columns for customers: {missing_customers}')
missing_stores = set(stores) - set(df_cost['store'])
if missing_stores:
    raise ValueError(f'Missing cost rows for stores: {missing_stores}')
cost = {}
for (_, row) in df_cost.iterrows():
    s = row['store']
    for c in customers:
        cost[s, c] = float(row[c])
if set(customers) != set(cost_columns):
    raise ValueError('Mismatch between customer demand and cost columns.')
if set(stores) != set(df_cost['store']):
    raise ValueError('Mismatch between supply capacity and cost rows.')
m = gp.Model('Walmart_Transportation')
x = m.addVars(stores, customers, vtype=gp.GRB.CONTINUOUS, lb=0.0, name='')
m.setObjective(gp.quicksum((cost[s, c] * x[s, c] for s in stores for c in customers)), gp.GRB.MINIMIZE)
for c in customers:
    m.addConstr(gp.quicksum((x[s, c] for s in stores)) == demand[c], name=f'demand_{c}')
for s in stores:
    m.addConstr(gp.quicksum((x[s, c] for c in customers)) <= supply_capacity[s], name=f'supply_{s}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total transportation cost: {m.objVal:.6f}')
    print('--- Optimal Transportation Plan (quantities shipped) ---')
    for s in stores:
        for c in customers:
            val = x[s, c].X
            if val > 1e-06:
                print(f'  Store {s} -> Customer {c}: {val:.6f}')
else:
    print(f'No optimal solution found. Status: {m.status}')