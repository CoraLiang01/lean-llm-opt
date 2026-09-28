import gurobipy as gp
import pandas as pd
import numpy as np
import re
customer_demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP6/customer_demand.csv'
supply_capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP6/supply_capacity.csv'
transportation_costs_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP6/transportation_costs.csv'
df_demand = pd.read_csv(customer_demand_path, sep=',')
df_demand['customer'] = df_demand['customer'].astype(str).str.strip()
customers = list(df_demand['customer'])
demand_dict = dict(zip(df_demand['customer'], df_demand['demand']))
df_supply = pd.read_csv(supply_capacity_path, sep=',')
df_supply['warehouse'] = df_supply['Unnamed: 0'].astype(str).str.strip()
warehouses = list(df_supply['warehouse'])
supply_dict = dict(zip(df_supply['warehouse'], df_supply['supply_capacity']))
df_cost = pd.read_csv(transportation_costs_path, sep=',')
df_cost['warehouse'] = df_cost['Unnamed: 0'].astype(str).str.strip()
cost_customer_cols = [col for col in df_cost.columns if col.startswith('C')]
cost_matrix = {}
for idx, row in df_cost.iterrows():
    s = str(row['warehouse']).strip()
    cost_matrix[s] = {}
    for c in customers:
        if c not in df_cost.columns:
            raise KeyError(f"Customer '{c}' not found in transportation_costs.csv columns.")
        cost_matrix[s][c] = float(row[c])
if set(warehouses) != set(cost_matrix.keys()):
    raise ValueError('Mismatch between warehouses in supply_capacity.csv and transportation_costs.csv.')
if set(customers) != set(cost_customer_cols):
    raise ValueError('Mismatch between customers in customer_demand.csv and transportation_costs.csv.')
m = gp.Model('TransportationOptimization')
x = m.addVars(warehouses, customers, vtype=gp.GRB.CONTINUOUS, lb=0.0, name='')
m.setObjective(gp.quicksum((cost_matrix[s][c] * x[s, c] for s in warehouses for c in customers)), gp.GRB.MINIMIZE)
for c in customers:
    m.addConstr(gp.quicksum((x[s, c] for s in warehouses)) == demand_dict[c], name='')
for s in warehouses:
    m.addConstr(gp.quicksum((x[s, c] for c in customers)) <= supply_dict[s], name='')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total transportation cost: {m.objVal:.2f}')
    print('--- Optimal Shipment Plan (nonzero shipments only) ---')
    for s in warehouses:
        for c in customers:
            shipped = x[s, c].X
            if shipped > 1e-06:
                print(f'Warehouse {s} -> Customer {c}: {shipped:.2f} units')
else:
    print(f'No optimal solution found. Status: {m.status}')