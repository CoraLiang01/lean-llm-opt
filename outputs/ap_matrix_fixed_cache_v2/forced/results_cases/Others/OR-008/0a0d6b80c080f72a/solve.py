import gurobipy as gp
import pandas as pd
import numpy as np
import re
customer_demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP8/customer_demand.csv'
supply_capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP8/supply_capacity.csv'
transportation_costs_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP8/transportation_costs.csv'
customer_demand_df = pd.read_csv(customer_demand_path, dtype=str, keep_default_na=False)
if not {'Customers', 'demand'}.issubset(customer_demand_df.columns):
    raise KeyError("customer_demand.csv must contain columns 'Customers' and 'demand'.")
customer_demand_df['Customers'] = customer_demand_df['Customers'].str.strip()
customers = customer_demand_df['Customers'].tolist()
customer_demand_df['demand'] = customer_demand_df['demand'].astype(int)
demand_dict = dict(zip(customer_demand_df['Customers'], customer_demand_df['demand']))
supply_capacity_df = pd.read_csv(supply_capacity_path, dtype=str, keep_default_na=False)
if not {'Suppliers', 'supply_capacity'}.issubset(supply_capacity_df.columns):
    raise KeyError("supply_capacity.csv must contain columns 'Suppliers' and 'supply_capacity'.")
supply_capacity_df['Suppliers'] = supply_capacity_df['Suppliers'].str.strip()
suppliers = supply_capacity_df['Suppliers'].tolist()
supply_capacity_df['supply_capacity'] = supply_capacity_df['supply_capacity'].astype(int)
supply_capacity_dict = dict(zip(supply_capacity_df['Suppliers'], supply_capacity_df['supply_capacity']))
transportation_costs_df = pd.read_csv(transportation_costs_path, dtype=str, keep_default_na=False)
if 'Unnamed: 0' not in transportation_costs_df.columns:
    raise KeyError("transportation_costs.csv must contain column 'Unnamed: 0' for supplier IDs.")
transportation_costs_df['Unnamed: 0'] = transportation_costs_df['Unnamed: 0'].str.strip()
cost_suppliers = transportation_costs_df['Unnamed: 0'].tolist()
cost_customers = [col for col in transportation_costs_df.columns if col != 'Unnamed: 0']
if set(suppliers) != set(cost_suppliers):
    raise ValueError(f'Supplier mismatch between supply_capacity.csv and transportation_costs.csv: {set(suppliers)} vs {set(cost_suppliers)}')
if set(customers) != set(cost_customers):
    raise ValueError(f'Customer mismatch between customer_demand.csv and transportation_costs.csv: {set(customers)} vs {set(cost_customers)}')
cost_dict = {}
for _, row in transportation_costs_df.iterrows():
    s = row['Unnamed: 0']
    for c in customers:
        val = row[c]
        try:
            cost_dict[s, c] = int(val)
        except Exception:
            raise ValueError(f"Invalid cost value for supplier '{s}', customer '{c}': '{val}'")
m = gp.Model('FreshMart_Transportation')
x_vars = m.addVars(suppliers, customers, vtype=gp.GRB.CONTINUOUS, lb=0.0, name='')
m.setObjective(gp.quicksum((cost_dict[s, c] * x_vars[s, c] for s in suppliers for c in customers)), gp.GRB.MINIMIZE)
for s in suppliers:
    m.addConstr(gp.quicksum((x_vars[s, c] for c in customers)) <= supply_capacity_dict[s])
for c in customers:
    m.addConstr(gp.quicksum((x_vars[s, c] for s in suppliers)) == demand_dict[c])
m.optimize()