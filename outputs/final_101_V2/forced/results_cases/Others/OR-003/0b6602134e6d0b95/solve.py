import gurobipy as gp
import pandas as pd
import numpy as np
import re
customer_demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP3/customer_demand.csv'
supply_capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP3/supply_capacity.csv'
transportation_costs_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP3/transportation_costs.csv'
df_demand = pd.read_csv(customer_demand_path, sep=',')
df_demand['customer'] = df_demand['customer'].astype(str).str.strip()
customers = list(df_demand['customer'])
demand_dict = dict(zip(df_demand['customer'], df_demand['demand']))
df_supply = pd.read_csv(supply_capacity_path, sep=',')
df_supply['supplier'] = df_supply['Unnamed: 0'].astype(str).str.strip()
suppliers = list(df_supply['supplier'])
supply_capacity_dict = dict(zip(df_supply['supplier'], df_supply['supply_capacity']))
df_cost = pd.read_csv(transportation_costs_path, sep=',')
df_cost['supplier'] = df_cost['Unnamed: 0'].astype(str).str.strip()
cost_customer_cols = [c for c in df_cost.columns if c in customers]
if set(cost_customer_cols) != set(customers):
    missing = set(customers) - set(cost_customer_cols)
    raise ValueError(f'Missing transportation cost columns for customers: {missing}')
if set(df_cost['supplier']) != set(suppliers):
    missing = set(suppliers) - set(df_cost['supplier'])
    raise ValueError(f'Missing transportation cost rows for suppliers: {missing}')
cost = {}
for _, row in df_cost.iterrows():
    s = str(row['supplier']).strip()
    for c in customers:
        cost[s, c] = float(row[c])
if set(demand_dict.keys()) != set(customers):
    raise ValueError('Mismatch in customer demand keys and customer list.')
if set(supply_capacity_dict.keys()) != set(suppliers):
    raise ValueError('Mismatch in supply capacity keys and supplier list.')
m = gp.Model('TransportationProblem')
x = m.addVars(suppliers, customers, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((cost[s, c] * x[s, c] for s in suppliers for c in customers)), gp.GRB.MINIMIZE)
for s in suppliers:
    m.addConstr(gp.quicksum((x[s, c] for c in customers)) <= supply_capacity_dict[s], name=f'supply_{s}')
for c in customers:
    m.addConstr(gp.quicksum((x[s, c] for s in suppliers)) == demand_dict[c], name=f'demand_{c}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total transportation cost: {m.objVal:.2f}')
    print('\n--- Optimal Transportation Plan (amount shipped from each supplier to each customer) ---')
    for s in suppliers:
        for c in customers:
            shipped = x[s, c].X
            if shipped > 1e-06:
                print(f'  Supplier {s} -> Customer {c}: {shipped:.2f}')
    print('\n--- Supplier Utilization ---')
    for s in suppliers:
        total_out = sum((x[s, c].X for c in customers))
        print(f'  Supplier {s}: {total_out:.2f} / {supply_capacity_dict[s]}')
    print('\n--- Customer Demand Satisfaction ---')
    for c in customers:
        total_in = sum((x[s, c].X for s in suppliers))
        print(f'  Customer {c}: {total_in:.2f} / {demand_dict[c]}')
else:
    print(f'No optimal solution found. Status: {m.status}')