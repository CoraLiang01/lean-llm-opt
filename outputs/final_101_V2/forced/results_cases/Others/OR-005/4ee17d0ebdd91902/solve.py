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
demand = dict(zip(df_demand['Customers'], df_demand['demand']))
df_supply = pd.read_csv(supply_capacity_path, sep=',')
df_supply['Supplier'] = df_supply['Supplier'].astype(str).str.strip()
suppliers = df_supply['Supplier'].tolist()
supply_capacity = dict(zip(df_supply['Supplier'], df_supply['supply_capacity']))
df_cost = pd.read_csv(transportation_costs_path, sep=',')
df_cost['Unnamed: 0'] = df_cost['Unnamed: 0'].astype(str).str.strip()
cost_suppliers = df_cost['Unnamed: 0'].tolist()
cost_customers = [col for col in df_cost.columns if col != 'Unnamed: 0']
missing_suppliers = set(suppliers) - set(cost_suppliers)
if missing_suppliers:
    raise ValueError(f'Missing suppliers in transportation_costs.csv: {missing_suppliers}')
missing_customers = set(customers) - set(cost_customers)
if missing_customers:
    raise ValueError(f'Missing customers in transportation_costs.csv: {missing_customers}')
c = {}
for i, row in df_cost.iterrows():
    supplier = row['Unnamed: 0']
    for customer in customers:
        c[supplier, customer] = float(row[customer])
m = gp.Model('TransportationProblem')
x = m.addVars(suppliers, customers, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((c[s, d] * x[s, d] for s in suppliers for d in customers)), gp.GRB.MINIMIZE)
for d in customers:
    m.addConstr(gp.quicksum((x[s, d] for s in suppliers)) == demand[d], name=f'demand_{d}')
for s in suppliers:
    m.addConstr(gp.quicksum((x[s, d] for d in customers)) <= supply_capacity[s], name=f'supply_{s}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total transportation cost: {m.objVal:.6f}')
    print('--- Optimal Shipment Plan (units shipped from each supplier to each customer) ---')
    for s in suppliers:
        for d in customers:
            val = x[s, d].X
            if val > 1e-06:
                print(f'  {s} -> {d}: {val:.6f}')
else:
    print(f'No optimal solution found. Status: {m.status}')