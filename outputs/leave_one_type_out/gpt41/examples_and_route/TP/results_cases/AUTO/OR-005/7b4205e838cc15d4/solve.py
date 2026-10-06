import gurobipy as gp
import pandas as pd
import numpy as np
import re
customer_demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP5/customer_demand.csv'
supply_capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP5/supply_capacity.csv'
transportation_costs_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP5/transportation_costs.csv'
df_demand = pd.read_csv(customer_demand_path, sep=',')
df_demand['Customers'] = df_demand['Customers'].astype(str).str.strip()
customers = df_demand['Customers'].tolist()
demand_dict = dict(zip(df_demand['Customers'], df_demand['demand']))
df_supply = pd.read_csv(supply_capacity_path, sep=',')
df_supply['Supplier'] = df_supply['Supplier'].astype(str).str.strip()
suppliers = df_supply['Supplier'].tolist()
supply_capacity_dict = dict(zip(df_supply['Supplier'], df_supply['supply_capacity']))
df_costs = pd.read_csv(transportation_costs_path, sep=',')
df_costs['Unnamed: 0'] = df_costs['Unnamed: 0'].astype(str).str.strip()
costs_suppliers = df_costs['Unnamed: 0'].tolist()
costs_customers = [col for col in df_costs.columns if col != 'Unnamed: 0']
if set(suppliers) != set(costs_suppliers):
    raise ValueError(f'Supplier mismatch between supply_capacity.csv and transportation_costs.csv: {set(suppliers) ^ set(costs_suppliers)}')
if set(customers) != set(costs_customers):
    raise ValueError(f'Customer mismatch between customer_demand.csv and transportation_costs.csv: {set(customers) ^ set(costs_customers)}')
cost = {}
for i, row in df_costs.iterrows():
    supplier = row['Unnamed: 0']
    for customer in customers:
        val = row[customer]
        if pd.isnull(val):
            raise ValueError(f'Missing transportation cost for supplier {supplier}, customer {customer}')
        cost[supplier, customer] = float(val)
m = gp.Model('TransportationProblem')
x = m.addVars(suppliers, customers, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((cost[i, j] * x[i, j] for i in suppliers for j in customers)), gp.GRB.MINIMIZE)
for j in customers:
    m.addConstr(gp.quicksum((x[i, j] for i in suppliers)) == demand_dict[j], name=f'demand_{j}')
for i in suppliers:
    m.addConstr(gp.quicksum((x[i, j] for j in customers)) <= supply_capacity_dict[i], name=f'supply_{i}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total transportation cost: {m.objVal:.6f}')
    print('--- Optimal Shipment Plan (quantities shipped from each supplier to each customer) ---')
    for i in suppliers:
        for j in customers:
            shipped = x[i, j].X
            if shipped > 1e-06:
                print(f'  {i} -> {j}: {shipped:.2f}')
else:
    print(f'No optimal solution found. Status: {m.status}')