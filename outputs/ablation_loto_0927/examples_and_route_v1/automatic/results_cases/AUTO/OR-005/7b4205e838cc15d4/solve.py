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
cost_supplier_names = df_costs['Unnamed: 0'].tolist()
if set(cost_supplier_names) != set(suppliers):
    raise ValueError(f'Supplier names in transportation_costs.csv do not match those in supply_capacity.csv.\nSupply file: {suppliers}\nCost file: {cost_supplier_names}')
cost_customer_names = [col for col in df_costs.columns if col != 'Unnamed: 0']
if set(cost_customer_names) != set(customers):
    raise ValueError(f'Customer names in transportation_costs.csv do not match those in customer_demand.csv.\nDemand file: {customers}\nCost file: {cost_customer_names}')
cost_dict = {}
for (idx, row) in df_costs.iterrows():
    s = row['Unnamed: 0']
    cost_dict[s] = {}
    for c in customers:
        val = row[c]
        if pd.isnull(val):
            raise ValueError(f'Missing transportation cost for supplier {s}, customer {c}')
        cost_dict[s][c] = float(val)
m = gp.Model('TransportationProblem')
x = m.addVars(suppliers, customers, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((cost_dict[s][c] * x[s, c] for s in suppliers for c in customers)), gp.GRB.MINIMIZE)
for c in customers:
    m.addConstr(gp.quicksum((x[s, c] for s in suppliers)) == demand_dict[c], name=f'demand_{c}')
for s in suppliers:
    m.addConstr(gp.quicksum((x[s, c] for c in customers)) <= supply_capacity_dict[s], name=f'supply_{s}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total transportation cost: {m.objVal:.6f}')
    print('--- Optimal Shipment Plan (quantities shipped from each supplier to each customer) ---')
    for s in suppliers:
        for c in customers:
            shipped = x[s, c].X
            if shipped > 1e-06:
                print(f'  {s} -> {c}: {shipped:.6f}')
else:
    print(f'No optimal solution found. Status: {m.status}')