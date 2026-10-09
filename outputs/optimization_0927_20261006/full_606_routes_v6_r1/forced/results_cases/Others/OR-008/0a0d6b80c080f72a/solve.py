import gurobipy as gp
import pandas as pd
import numpy as np
import re
customer_demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP8/customer_demand.csv'
supply_capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP8/supply_capacity.csv'
transportation_costs_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP8/transportation_costs.csv'
customer_demand_df = pd.read_csv(customer_demand_path, dtype=str, keep_default_na=False)
supply_capacity_df = pd.read_csv(supply_capacity_path, dtype=str, keep_default_na=False)
transportation_costs_df = pd.read_csv(transportation_costs_path, dtype=str, keep_default_na=False)
if 'Customers' not in customer_demand_df.columns or 'demand' not in customer_demand_df.columns:
    raise KeyError("customer_demand.csv must contain 'Customers' and 'demand' columns.")
customers = customer_demand_df['Customers'].astype(str).str.strip()
customer_list = customers.tolist()
demand_series = customer_demand_df.set_index(customer_demand_df['Customers'].astype(str).str.strip())['demand'].astype(int)
demand_dict = demand_series.to_dict()
if 'Suppliers' not in supply_capacity_df.columns or 'supply_capacity' not in supply_capacity_df.columns:
    raise KeyError("supply_capacity.csv must contain 'Suppliers' and 'supply_capacity' columns.")
suppliers = supply_capacity_df['Suppliers'].astype(str).str.strip()
supplier_list = suppliers.tolist()
supply_series = supply_capacity_df.set_index(supply_capacity_df['Suppliers'].astype(str).str.strip())['supply_capacity'].astype(int)
supply_capacity_dict = supply_series.to_dict()
if 'Unnamed: 0' not in transportation_costs_df.columns:
    raise KeyError("transportation_costs.csv must contain 'Unnamed: 0' column for suppliers.")
transportation_costs_df['Unnamed: 0'] = transportation_costs_df['Unnamed: 0'].astype(str).str.strip()
cost_supplier_list = transportation_costs_df['Unnamed: 0'].tolist()
cost_customer_cols = [col for col in transportation_costs_df.columns if col != 'Unnamed: 0']
missing_customers = set(customer_list) - set(cost_customer_cols)
if missing_customers:
    raise KeyError(f'transportation_costs.csv is missing columns for customers: {missing_customers}')
missing_suppliers = set(supplier_list) - set(cost_supplier_list)
if missing_suppliers:
    raise KeyError(f'transportation_costs.csv is missing rows for suppliers: {missing_suppliers}')
cost_dict = {}
for (idx, row) in transportation_costs_df.iterrows():
    supplier = str(row['Unnamed: 0']).strip()
    for customer in customer_list:
        val = row[customer]
        try:
            cost = int(val)
        except Exception:
            raise ValueError(f"Invalid transportation cost for supplier '{supplier}', customer '{customer}': '{val}'")
        cost_dict[supplier, customer] = cost
for s in supplier_list:
    for c in customer_list:
        if (s, c) not in cost_dict:
            raise KeyError(f"Missing transportation cost for supplier '{s}', customer '{c}'.")
m = gp.Model('FreshMart_Transportation')
x_vars = m.addVars(supplier_list, customer_list, vtype=gp.GRB.CONTINUOUS, lb=0.0, name='')
m.setObjective(gp.quicksum((cost_dict[s, c] * x_vars[s, c] for s in supplier_list for c in customer_list)), sense=gp.GRB.MINIMIZE)
for s in supplier_list:
    m.addConstr(gp.quicksum((x_vars[s, c] for c in customer_list)) <= supply_capacity_dict[s], name=f'supply_capacity_{s}')
for c in customer_list:
    m.addConstr(gp.quicksum((x_vars[s, c] for s in supplier_list)) == demand_dict[c], name=f'demand_{c}')
m.optimize()