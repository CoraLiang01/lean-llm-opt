import gurobipy as gp
import pandas as pd
import numpy as np
import re

def norm_id(x):
    return str(x).strip()
customer_demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP8/customer_demand.csv'
supply_capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP8/supply_capacity.csv'
transportation_costs_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP8/transportation_costs.csv'
df_cust = pd.read_csv(customer_demand_path, sep=',')
df_cust['Customers'] = df_cust['Customers'].apply(norm_id)
customers = list(df_cust['Customers'])
demand = dict(zip(df_cust['Customers'], df_cust['demand']))
df_supp = pd.read_csv(supply_capacity_path, sep=',')
df_supp['Suppliers'] = df_supp['Suppliers'].apply(norm_id)
suppliers = list(df_supp['Suppliers'])
supply_capacity = dict(zip(df_supp['Suppliers'], df_supp['supply_capacity']))
df_cost = pd.read_csv(transportation_costs_path, sep=',')
df_cost['Unnamed: 0'] = df_cost['Unnamed: 0'].apply(norm_id)
cost_suppliers = list(df_cost['Unnamed: 0'])
cost_customers = [norm_id(c) for c in df_cost.columns if c != 'Unnamed: 0']
if set(suppliers) != set(cost_suppliers):
    raise ValueError(f'Supplier mismatch between supply_capacity.csv and transportation_costs.csv: {set(suppliers)} vs {set(cost_suppliers)}')
if set(customers) != set(cost_customers):
    raise ValueError(f'Customer mismatch between customer_demand.csv and transportation_costs.csv: {set(customers)} vs {set(cost_customers)}')
cost = {}
for i, s in enumerate(cost_suppliers):
    for c in cost_customers:
        cost[s, c] = float(df_cost.loc[i, c])
m = gp.Model('FreshMart_Transportation')
x = m.addVars(suppliers, customers, vtype=gp.GRB.CONTINUOUS, lb=0.0, name='')
m.setObjective(gp.quicksum((cost[s, c] * x[s, c] for s in suppliers for c in customers)), sense=gp.GRB.MINIMIZE)
for s in suppliers:
    m.addConstr(gp.quicksum((x[s, c] for c in customers)) <= supply_capacity[s], name=f'supply_{s}')
for c in customers:
    m.addConstr(gp.quicksum((x[s, c] for s in suppliers)) == demand[c], name=f'demand_{c}')
m.optimize()