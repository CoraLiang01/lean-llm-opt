import gurobipy as gp
import pandas as pd
import numpy as np
import re
customer_demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP7/customer_demand.csv'
supply_capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP7/supply_capacity.csv'
transportation_costs_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP7/transportation_costs.csv'
customer_demand_df = pd.read_csv(customer_demand_path, sep=',')
supply_capacity_df = pd.read_csv(supply_capacity_path, sep=',')
transportation_costs_df = pd.read_csv(transportation_costs_path, sep=',')
warehouses = supply_capacity_df['region'].astype(str).str.strip().tolist()
stores = customer_demand_df['customer'].astype(str).str.strip().tolist()
cost_row_ids = transportation_costs_df['Unnamed: 0'].astype(str).str.strip().tolist()
if set(cost_row_ids) != set(warehouses):
    raise ValueError(f'Mismatch between warehouses in supply_capacity.csv and transportation_costs.csv rows: {set(warehouses)} vs {set(cost_row_ids)}')
cost_col_ids = [col for col in transportation_costs_df.columns if col != 'Unnamed: 0']
if set(cost_col_ids) != set(stores):
    raise ValueError(f'Mismatch between stores in customer_demand.csv and transportation_costs.csv columns: {set(stores)} vs {set(cost_col_ids)}')
cost = {}
for i, w in enumerate(cost_row_ids):
    for s in stores:
        cost[w, s] = float(transportation_costs_df.loc[i, s])
demand = {}
for _, row in customer_demand_df.iterrows():
    s = str(row['customer']).strip()
    demand[s] = float(row['demand'])
supply_capacity = {}
for _, row in supply_capacity_df.iterrows():
    w = str(row['region']).strip()
    supply_capacity[w] = float(row['supply_capacity'])
if set(demand.keys()) != set(stores):
    raise ValueError('Mismatch in store identifiers between demand and stores list.')
if set(supply_capacity.keys()) != set(warehouses):
    raise ValueError('Mismatch in warehouse identifiers between supply_capacity and warehouses list.')
m = gp.Model('GreenMart_Transportation')
x = m.addVars(warehouses, stores, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((cost[w, s] * x[w, s] for w in warehouses for s in stores)), sense=gp.GRB.MINIMIZE)
for s in stores:
    m.addConstr(gp.quicksum((x[w, s] for w in warehouses)) == demand[s], name=f'demand_{s}')
for w in warehouses:
    m.addConstr(gp.quicksum((x[w, s] for s in stores)) <= supply_capacity[w], name=f'supply_{w}')
m.optimize()