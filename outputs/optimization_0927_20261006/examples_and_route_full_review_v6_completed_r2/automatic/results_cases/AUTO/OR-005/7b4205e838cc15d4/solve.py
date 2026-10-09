import gurobipy as gp
import pandas as pd
import numpy as np
import re
customer_demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP5/customer_demand.csv'
supply_capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP5/supply_capacity.csv'
transportation_costs_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP5/transportation_costs.csv'
customer_demand_df = pd.read_csv(customer_demand_path, dtype=str, keep_default_na=False)
if 'Customers' not in customer_demand_df.columns or 'demand' not in customer_demand_df.columns:
    raise KeyError("customer_demand.csv must contain 'Customers' and 'demand' columns.")
customer_demand_df['Customers'] = customer_demand_df['Customers'].str.strip()
customer_demand_df['demand'] = customer_demand_df['demand'].astype(float)
customers = customer_demand_df['Customers'].tolist()
demand_dict = dict(zip(customer_demand_df['Customers'], customer_demand_df['demand']))
supply_capacity_df = pd.read_csv(supply_capacity_path, dtype=str, keep_default_na=False)
if 'Supplier' not in supply_capacity_df.columns or 'supply_capacity' not in supply_capacity_df.columns:
    raise KeyError("supply_capacity.csv must contain 'Supplier' and 'supply_capacity' columns.")
supply_capacity_df['Supplier'] = supply_capacity_df['Supplier'].str.strip()
supply_capacity_df['supply_capacity'] = supply_capacity_df['supply_capacity'].astype(float)
suppliers = supply_capacity_df['Supplier'].tolist()
supply_capacity_dict = dict(zip(supply_capacity_df['Supplier'], supply_capacity_df['supply_capacity']))
transportation_costs_df = pd.read_csv(transportation_costs_path, dtype=str, keep_default_na=False)
if 'Unnamed: 0' not in transportation_costs_df.columns:
    raise KeyError("transportation_costs.csv must contain 'Unnamed: 0' column for supplier IDs.")
transportation_costs_df['Unnamed: 0'] = transportation_costs_df['Unnamed: 0'].str.strip()
for cust in customers:
    if cust not in transportation_costs_df.columns:
        raise KeyError(f"Customer '{cust}' not found as a column in transportation_costs.csv.")
cost_dict = {}
for (idx, row) in transportation_costs_df.iterrows():
    supplier_id = row['Unnamed: 0']
    if supplier_id not in suppliers:
        raise KeyError(f"Supplier '{supplier_id}' in transportation_costs.csv not found in supply_capacity.csv.")
    for cust in customers:
        try:
            cost = float(row[cust])
        except Exception as e:
            raise ValueError(f"Invalid cost value for supplier '{supplier_id}', customer '{cust}': {row[cust]}")
        cost_dict[supplier_id, cust] = cost
for i in suppliers:
    for j in customers:
        if (i, j) not in cost_dict:
            raise KeyError(f"Missing transportation cost for supplier '{i}', customer '{j}'.")
m = gp.Model('TransportationProblem')
x_vars = m.addVars(suppliers, customers, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((cost_dict[i, j] * x_vars[i, j] for i in suppliers for j in customers)), gp.GRB.MINIMIZE)
for j in customers:
    m.addConstr(gp.quicksum((x_vars[i, j] for i in suppliers)) == demand_dict[j], name=f'demand_{j}')
for i in suppliers:
    m.addConstr(gp.quicksum((x_vars[i, j] for j in customers)) <= supply_capacity_dict[i], name=f'supply_{i}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total transportation cost: {m.objVal:.6f}')
    print('--- Optimal Shipment Plan (units shipped from each supplier to each customer) ---')
    for i in suppliers:
        for j in customers:
            val = x_vars[i, j].X
            if val > 1e-06:
                print(f'  Supplier {i} -> Customer {j}: {val:.6f}')
else:
    print(f'No optimal solution found. Status: {m.status}')