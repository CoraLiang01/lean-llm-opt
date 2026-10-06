import gurobipy as gp
import pandas as pd
import numpy as np
customer_demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP4/customer_demand.csv'
supply_capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP4/supply_capacity.csv'
transportation_costs_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP4/transportation_costs.csv'
df_demand = pd.read_csv(customer_demand_path, sep=',')
df_demand['customer'] = df_demand['customer'].astype(str).str.strip()
customers = df_demand['customer'].tolist()
demand = dict(zip(df_demand['customer'], df_demand['demand']))
df_supply = pd.read_csv(supply_capacity_path, sep=',')
df_supply['Unnamed: 0'] = df_supply['Unnamed: 0'].astype(str).str.strip()
sources = df_supply['Unnamed: 0'].tolist()
supply_capacity = dict(zip(df_supply['Unnamed: 0'], df_supply['supply_capacity']))
df_cost = pd.read_csv(transportation_costs_path, sep=',')
df_cost['Unnamed: 0'] = df_cost['Unnamed: 0'].astype(str).str.strip()
cost_sources = df_cost['Unnamed: 0'].tolist()
cost_customers = [col for col in df_cost.columns if col != 'Unnamed: 0']
if set(sources) != set(cost_sources):
    raise ValueError(f'Mismatch in sources between supply_capacity.csv and transportation_costs.csv: {set(sources) ^ set(cost_sources)}')
if set(customers) != set(cost_customers):
    raise ValueError(f'Mismatch in customers between customer_demand.csv and transportation_costs.csv: {set(customers) ^ set(cost_customers)}')
cost = {}
for idx, row in df_cost.iterrows():
    s = row['Unnamed: 0']
    for c in customers:
        cost[s, c] = float(row[c])
m = gp.Model('TransportationOptimization')
x = m.addVars(sources, customers, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((cost[s, c] * x[s, c] for s in sources for c in customers)), gp.GRB.MINIMIZE)
for c in customers:
    m.addConstr(gp.quicksum((x[s, c] for s in sources)) == demand[c], name=f'demand_{c}')
for s in sources:
    m.addConstr(gp.quicksum((x[s, c] for c in customers)) <= supply_capacity[s], name=f'supply_{s}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total transportation cost: {m.objVal:.2f}')
    print('\n--- Optimal Shipping Plan (amount shipped from each source to each customer) ---')
    for s in sources:
        for c in customers:
            shipped = x[s, c].X
            if shipped > 1e-06:
                print(f'  From {s} to {c}: {shipped:.2f}')
    print('\n--- Source Utilization ---')
    for s in sources:
        total_out = sum((x[s, c].X for c in customers))
        print(f'  {s}: {total_out:.2f} / {supply_capacity[s]}')
    print('\n--- Customer Fulfillment ---')
    for c in customers:
        total_in = sum((x[s, c].X for s in sources))
        print(f'  {c}: {total_in:.2f} / {demand[c]}')
else:
    print(f'No optimal solution found. Status: {m.status}')