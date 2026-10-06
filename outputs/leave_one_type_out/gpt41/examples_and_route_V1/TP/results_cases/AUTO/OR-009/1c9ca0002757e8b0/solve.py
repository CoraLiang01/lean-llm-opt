import gurobipy as gp
import pandas as pd
import numpy as np
import re
customer_demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP9/customer_demand.csv'
supply_capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP9/supply_capacity.csv'
transportation_costs_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP9/transportation_costs.csv'
df_demand = pd.read_csv(customer_demand_path, sep=',')
df_demand['customer'] = df_demand['customer'].astype(str).str.strip()
customers = df_demand['customer'].tolist()
customer_demand = dict(zip(df_demand['customer'], df_demand['demand']))
df_supply = pd.read_csv(supply_capacity_path, sep=',')
df_supply['plant'] = df_supply['Unnamed: 0'].astype(str).str.strip()
plants = df_supply['plant'].tolist()
supply_capacity = dict(zip(df_supply['plant'], df_supply['supply_capacity']))
df_costs = pd.read_csv(transportation_costs_path, sep=',')
df_costs['plant'] = df_costs['Unnamed: 0'].astype(str).str.strip()
cost_plants = df_costs['plant'].tolist()
cost_customers = [col for col in df_costs.columns if col not in ['Unnamed: 0', 'plant']]
if set(plants) != set(cost_plants):
    raise ValueError(f'Mismatch between plants in supply_capacity and transportation_costs: {set(plants)} vs {set(cost_plants)}')
if set(customers) != set(cost_customers):
    raise ValueError(f'Mismatch between customers in customer_demand and transportation_costs: {set(customers)} vs {set(cost_customers)}')
cost = {}
for _, row in df_costs.iterrows():
    s = row['plant']
    for c in customers:
        cost[s, c] = float(row[c])
m = gp.Model('BrewCo_Transportation')
x = m.addVars(plants, customers, lb=0.0, name='')
m.setObjective(gp.quicksum((cost[s, c] * x[s, c] for s in plants for c in customers)), gp.GRB.MINIMIZE)
for c in customers:
    m.addConstr(gp.quicksum((x[s, c] for s in plants)) == customer_demand[c], name='')
for s in plants:
    m.addConstr(gp.quicksum((x[s, c] for c in customers)) <= supply_capacity[s], name='')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total transportation cost: {m.objVal:.2f}')
    print('--- Optimal Shipment Plan (quantities shipped from each plant to each customer) ---')
    for s in plants:
        for c in customers:
            shipped = x[s, c].X
            if shipped > 1e-06:
                print(f'  Plant {s} -> Customer {c}: {shipped:.2f}')
else:
    print(f'No optimal solution found. Status: {m.status}')