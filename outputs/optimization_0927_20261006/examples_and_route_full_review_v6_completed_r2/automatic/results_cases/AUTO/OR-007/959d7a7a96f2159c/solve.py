import gurobipy as gp
import pandas as pd
import numpy as np
import re
customer_demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP7/customer_demand.csv'
supply_capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP7/supply_capacity.csv'
transportation_costs_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP7/transportation_costs.csv'
customer_demand_df = pd.read_csv(customer_demand_path, dtype=str, keep_default_na=False)
supply_capacity_df = pd.read_csv(supply_capacity_path, dtype=str, keep_default_na=False)
transportation_costs_df = pd.read_csv(transportation_costs_path, dtype=str, keep_default_na=False)
if 'customer' not in customer_demand_df.columns or 'demand' not in customer_demand_df.columns:
    raise KeyError("customer_demand.csv must contain columns 'customer' and 'demand'")
customers = customer_demand_df['customer'].astype(str).str.strip()
customer_set = list(customers.unique())
if 'region' not in supply_capacity_df.columns or 'supply_capacity' not in supply_capacity_df.columns:
    raise KeyError("supply_capacity.csv must contain columns 'region' and 'supply_capacity'")
warehouses = supply_capacity_df['region'].astype(str).str.strip()
warehouse_set = list(warehouses.unique())
customer_demand_df['demand'] = customer_demand_df['demand'].astype(float)
demand_dict = dict(zip(customer_demand_df['customer'].astype(str).str.strip(), customer_demand_df['demand']))
supply_capacity_df['supply_capacity'] = supply_capacity_df['supply_capacity'].astype(float)
supply_capacity_dict = dict(zip(supply_capacity_df['region'].astype(str).str.strip(), supply_capacity_df['supply_capacity']))
if 'Unnamed: 0' not in transportation_costs_df.columns:
    raise KeyError("transportation_costs.csv must have 'Unnamed: 0' as warehouse identifier column")
cost_matrix = {}
for (idx, row) in transportation_costs_df.iterrows():
    warehouse_id = str(row['Unnamed: 0']).strip()
    if warehouse_id not in warehouse_set:
        continue
    for customer_id in customer_set:
        if customer_id not in transportation_costs_df.columns:
            raise KeyError(f"Customer '{customer_id}' not found as a column in transportation_costs.csv")
        cost_val = row[customer_id]
        try:
            cost_val = float(cost_val)
        except Exception:
            raise ValueError(f"Invalid cost value for warehouse '{warehouse_id}', customer '{customer_id}': {cost_val}")
        cost_matrix[warehouse_id, customer_id] = cost_val
for w in warehouse_set:
    for c in customer_set:
        if (w, c) not in cost_matrix:
            raise KeyError(f"Missing transportation cost for warehouse '{w}', customer '{c}'")
m = gp.Model('GreenMart_Transportation')
x_vars = m.addVars(warehouse_set, customer_set, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((cost_matrix[w, c] * x_vars[w, c] for w in warehouse_set for c in customer_set)), gp.GRB.MINIMIZE)
for c in customer_set:
    m.addConstr(gp.quicksum((x_vars[w, c] for w in warehouse_set)) == demand_dict[c], name=f'demand_{c}')
for w in warehouse_set:
    m.addConstr(gp.quicksum((x_vars[w, c] for c in customer_set)) <= supply_capacity_dict[w], name=f'supply_{w}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total transportation cost: {m.objVal:.2f}')
    print('--- Optimal Shipment Plan ---')
    for w in warehouse_set:
        for c in customer_set:
            shipped = x_vars[w, c].X
            if shipped > 1e-06:
                print(f'  Warehouse {w} -> Store {c}: {shipped:.2f} units')
else:
    print(f'No optimal solution found. Status: {m.status}')