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
demand = dict(zip(df_demand['customer'], df_demand['demand']))
df_supply = pd.read_csv(supply_capacity_path, sep=',')
df_supply['plant'] = df_supply['Unnamed: 0'].astype(str).str.strip()
plants = df_supply['plant'].tolist()
supply_capacity = dict(zip(df_supply['plant'], df_supply['supply_capacity']))
df_cost = pd.read_csv(transportation_costs_path, sep=',')
df_cost['plant'] = df_cost['Unnamed: 0'].astype(str).str.strip()
for c in customers:
    if c not in df_cost.columns:
        raise KeyError(f"Customer '{c}' not found in transportation_costs.csv columns.")
cost = {}
for _, row in df_cost.iterrows():
    s = str(row['plant']).strip()
    for c in customers:
        cost[s, c] = float(row[c])
for s in plants:
    for c in customers:
        if (s, c) not in cost:
            raise KeyError(f"Missing transportation cost for plant '{s}' and customer '{c}'.")
for s in plants:
    if s not in supply_capacity:
        raise KeyError(f"Plant '{s}' missing in supply_capacity.csv.")
for c in customers:
    if c not in demand:
        raise KeyError(f"Customer '{c}' missing in customer_demand.csv.")
m = gp.Model('BrewCo_Transportation')
x = m.addVars(plants, customers, lb=0.0, name='')
m.setObjective(gp.quicksum((cost[s, c] * x[s, c] for s in plants for c in customers)), gp.GRB.MINIMIZE)
for c in customers:
    m.addConstr(gp.quicksum((x[s, c] for s in plants)) == demand[c], name=f'demand_{c}')
for s in plants:
    m.addConstr(gp.quicksum((x[s, c] for c in customers)) <= supply_capacity[s], name=f'supply_{s}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total transportation cost: {m.objVal:.2f}')
    print('--- Optimal Shipment Plan (quantities shipped from each plant to each customer) ---')
    for s in plants:
        for c in customers:
            shipped = x[s, c].X
            if shipped > 1e-06:
                print(f'  Plant {s} -> Customer {c}: {shipped:.2f} units')
    print('\n--- Plant Utilization ---')
    for s in plants:
        total_out = sum((x[s, c].X for c in customers))
        print(f'  Plant {s}: {total_out:.2f} / {supply_capacity[s]} units used')
    print('\n--- Customer Demand Fulfillment ---')
    for c in customers:
        total_in = sum((x[s, c].X for s in plants))
        print(f'  Customer {c}: {total_in:.2f} / {demand[c]} units received')
else:
    print(f'No optimal solution found. Status: {m.status}')