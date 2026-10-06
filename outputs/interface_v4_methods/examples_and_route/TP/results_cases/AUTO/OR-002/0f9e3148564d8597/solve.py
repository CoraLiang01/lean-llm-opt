import gurobipy as gp
import pandas as pd
import numpy as np
customer_demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP2/customer_demand.csv'
supply_capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP2/supply_capacity.csv'
transportation_costs_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP2/transportation_costs.csv'
df_demand = pd.read_csv(customer_demand_path, sep=',')
df_demand['customer'] = df_demand['customer'].astype(str).str.strip()
customers = df_demand['customer'].tolist()
customer_demand = dict(zip(df_demand['customer'], df_demand['demand']))
df_supply = pd.read_csv(supply_capacity_path, sep=',')
df_supply['store'] = df_supply['Unnamed: 0'].astype(str).str.strip()
stores = df_supply['store'].tolist()
supply_capacity = dict(zip(df_supply['store'], df_supply['supply_capacity']))
df_cost = pd.read_csv(transportation_costs_path, sep=',')
df_cost['store'] = df_cost['Unnamed: 0'].astype(str).str.strip()
cost_stores = df_cost['store'].tolist()
cost_customers = [col for col in df_cost.columns if col.startswith('C')]
if set(stores) != set(cost_stores):
    raise ValueError('Mismatch between stores in supply_capacity.csv and transportation_costs.csv')
if set(customers) != set(cost_customers):
    raise ValueError('Mismatch between customers in customer_demand.csv and transportation_costs.csv')
cost = {}
for _, row in df_cost.iterrows():
    s = row['store']
    for c in customers:
        cost[s, c] = float(row[c])
for s in stores:
    for c in customers:
        if (s, c) not in cost:
            raise ValueError(f'Missing transportation cost for store {s}, customer {c}')
m = gp.Model('Walmart_Transportation')
x = m.addVars(stores, customers, vtype=gp.GRB.CONTINUOUS, lb=0.0, name='')
m.setObjective(gp.quicksum((cost[s, c] * x[s, c] for s in stores for c in customers)), sense=gp.GRB.MINIMIZE)
for c in customers:
    m.addConstr(gp.quicksum((x[s, c] for s in stores)) == customer_demand[c], name=f'demand_{c}')
for s in stores:
    m.addConstr(gp.quicksum((x[s, c] for c in customers)) <= supply_capacity[s], name=f'supply_{s}')
m.optimize()