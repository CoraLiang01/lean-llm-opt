import gurobipy as gp
import pandas as pd
import numpy as np
import re
customer_demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP7/customer_demand.csv'
supply_capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP7/supply_capacity.csv'
transportation_costs_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP7/transportation_costs.csv'
customer_demand_df = pd.read_csv(customer_demand_path, dtype=str, keep_default_na=False)
if not {'customer', 'demand'}.issubset(customer_demand_df.columns):
    raise KeyError("customer_demand.csv must contain columns 'customer' and 'demand'.")
customer_demand_df['customer'] = customer_demand_df['customer'].str.strip()
customers = customer_demand_df['customer'].tolist()
customer_demand_df['demand'] = customer_demand_df['demand'].astype(int)
demand_dict = dict(zip(customer_demand_df['customer'], customer_demand_df['demand']))
supply_capacity_df = pd.read_csv(supply_capacity_path, dtype=str, keep_default_na=False)
if not {'region', 'supply_capacity'}.issubset(supply_capacity_df.columns):
    raise KeyError("supply_capacity.csv must contain columns 'region' and 'supply_capacity'.")
supply_capacity_df['region'] = supply_capacity_df['region'].str.strip()
warehouses = supply_capacity_df['region'].tolist()
supply_capacity_df['supply_capacity'] = supply_capacity_df['supply_capacity'].astype(int)
supply_capacity_dict = dict(zip(supply_capacity_df['region'], supply_capacity_df['supply_capacity']))
transportation_costs_df = pd.read_csv(transportation_costs_path, dtype=str, keep_default_na=False)
if 'Unnamed: 0' not in transportation_costs_df.columns:
    raise KeyError("transportation_costs.csv must contain column 'Unnamed: 0' for warehouse IDs.")
transportation_costs_df['Unnamed: 0'] = transportation_costs_df['Unnamed: 0'].str.strip()
cost_warehouse_ids = transportation_costs_df['Unnamed: 0'].tolist()
cost_customer_ids = [col for col in transportation_costs_df.columns if col != 'Unnamed: 0']
missing_warehouses = set(warehouses) - set(cost_warehouse_ids)
if missing_warehouses:
    raise ValueError(f'Warehouses missing in transportation_costs.csv: {missing_warehouses}')
missing_customers = set(customers) - set(cost_customer_ids)
if missing_customers:
    raise ValueError(f'Customers missing in transportation_costs.csv: {missing_customers}')
cost_dict = {}
for idx, row in transportation_costs_df.iterrows():
    w = row['Unnamed: 0']
    for s in customers:
        val = row[s]
        try:
            cost_dict[w, s] = float(val)
        except Exception:
            raise ValueError(f'Invalid cost value for warehouse {w}, customer {s}: {val}')
m = gp.Model('GreenMart_Transportation')
x_vars = m.addVars(warehouses, customers, vtype=gp.GRB.CONTINUOUS, lb=0.0, name='')
m.setObjective(gp.quicksum((cost_dict[w, s] * x_vars[w, s] for w in warehouses for s in customers)), gp.GRB.MINIMIZE)
for s in customers:
    m.addConstr(gp.quicksum((x_vars[w, s] for w in warehouses)) == demand_dict[s], name=f'demand_{s}')
for w in warehouses:
    m.addConstr(gp.quicksum((x_vars[w, s] for s in customers)) <= supply_capacity_dict[w], name=f'supply_{w}')
m.optimize()