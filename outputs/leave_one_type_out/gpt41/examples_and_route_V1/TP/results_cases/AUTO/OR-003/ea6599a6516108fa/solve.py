import gurobipy as gp
import pandas as pd
import numpy as np
import re
customer_demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP3/customer_demand.csv'
supply_capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP3/supply_capacity.csv'
transportation_costs_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP3/transportation_costs.csv'
df_demand = pd.read_csv(customer_demand_path, sep=',')
df_demand['customer'] = df_demand['customer'].astype(str).str.strip()
customers = df_demand['customer'].tolist()
demand = dict(zip(df_demand['customer'], df_demand['demand']))
df_supply = pd.read_csv(supply_capacity_path, sep=',')
df_supply['supplier'] = df_supply['Unnamed: 0'].astype(str).str.strip()
suppliers = df_supply['supplier'].tolist()
supply_capacity = dict(zip(df_supply['supplier'], df_supply['supply_capacity']))
df_cost = pd.read_csv(transportation_costs_path, sep=',')
df_cost['supplier'] = df_cost['Unnamed: 0'].astype(str).str.strip()
cost_customer_cols = [c for c in customers if c in df_cost.columns]
if set(cost_customer_cols) != set(customers):
    missing = set(customers) - set(cost_customer_cols)
    raise KeyError(f'Missing transportation cost columns for customers: {missing}')
if set(df_cost['supplier']) != set(suppliers):
    missing = set(suppliers) - set(df_cost['supplier'])
    raise KeyError(f'Missing transportation cost rows for suppliers: {missing}')
cost = {}
for _, row in df_cost.iterrows():
    s = str(row['supplier']).strip()
    cost[s] = {}
    for c in customers:
        val = row[c]
        if pd.isnull(val):
            raise ValueError(f'Missing transportation cost for supplier {s}, customer {c}')
        cost[s][c] = float(val)
if set(demand.keys()) != set(customers):
    raise ValueError('Mismatch in customer demand keys and customer list')
if set(supply_capacity.keys()) != set(suppliers):
    raise ValueError('Mismatch in supply capacity keys and supplier list')
if set(cost.keys()) != set(suppliers):
    raise ValueError('Mismatch in cost keys and supplier list')
for s in suppliers:
    if set(cost[s].keys()) != set(customers):
        raise ValueError(f'Mismatch in cost columns for supplier {s}')
m = gp.Model('TransportationPlan')
x = m.addVars(suppliers, customers, lb=0.0, name='')
m.setObjective(gp.quicksum((cost[s][c] * x[s, c] for s in suppliers for c in customers)), gp.GRB.MINIMIZE)
for s in suppliers:
    m.addConstr(gp.quicksum((x[s, c] for c in customers)) <= supply_capacity[s], name=f'supply_{s}')
for c in customers:
    m.addConstr(gp.quicksum((x[s, c] for s in suppliers)) == demand[c], name=f'demand_{c}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total transportation cost: {m.objVal:.2f}')
    print('\n--- Optimal Transportation Plan (units shipped) ---')
    for s in suppliers:
        for c in customers:
            shipped = x[s, c].X
            if shipped > 1e-06:
                print(f'  Supplier {s} -> Customer {c}: {shipped:.2f}')
    print('\n--- Supplier Utilization ---')
    for s in suppliers:
        total_out = sum((x[s, c].X for c in customers))
        print(f'  Supplier {s}: {total_out:.2f} / {supply_capacity[s]}')
    print('\n--- Customer Demand Fulfillment ---')
    for c in customers:
        total_in = sum((x[s, c].X for s in suppliers))
        print(f'  Customer {c}: {total_in:.2f} / {demand[c]}')
else:
    print(f'No optimal solution found. Status: {m.status}')