import gurobipy as gp
import pandas as pd
import numpy as np
import re
customer_demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP5/customer_demand.csv'
supply_capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP5/supply_capacity.csv'
transportation_costs_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP5/transportation_costs.csv'
customer_demand_df = pd.read_csv(customer_demand_path, dtype=str, keep_default_na=False)
if not {'Customers', 'demand'}.issubset(customer_demand_df.columns):
    raise KeyError("customer_demand.csv must contain columns 'Customers' and 'demand'.")
customer_demand_df['Customers'] = customer_demand_df['Customers'].str.strip()
customers = customer_demand_df['Customers'].tolist()
customer_demand_df['demand'] = customer_demand_df['demand'].astype(int)
demand_dict = dict(zip(customer_demand_df['Customers'], customer_demand_df['demand']))
supply_capacity_df = pd.read_csv(supply_capacity_path, dtype=str, keep_default_na=False)
if not {'Supplier', 'supply_capacity'}.issubset(supply_capacity_df.columns):
    raise KeyError("supply_capacity.csv must contain columns 'Supplier' and 'supply_capacity'.")
supply_capacity_df['Supplier'] = supply_capacity_df['Supplier'].str.strip()
suppliers = supply_capacity_df['Supplier'].tolist()
supply_capacity_df['supply_capacity'] = supply_capacity_df['supply_capacity'].astype(int)
supply_capacity_dict = dict(zip(supply_capacity_df['Supplier'], supply_capacity_df['supply_capacity']))
transportation_costs_df = pd.read_csv(transportation_costs_path, dtype=str, keep_default_na=False)
if 'Unnamed: 0' not in transportation_costs_df.columns:
    raise KeyError("transportation_costs.csv must contain column 'Unnamed: 0' for supplier IDs.")
transportation_costs_df['Unnamed: 0'] = transportation_costs_df['Unnamed: 0'].str.strip()
cost_suppliers = transportation_costs_df['Unnamed: 0'].tolist()
cost_customers = [col for col in transportation_costs_df.columns if col != 'Unnamed: 0']
missing_suppliers = set(suppliers) - set(cost_suppliers)
if missing_suppliers:
    raise ValueError(f'Missing suppliers in transportation_costs.csv: {missing_suppliers}')
missing_customers = set(customers) - set(cost_customers)
if missing_customers:
    raise ValueError(f'Missing customers in transportation_costs.csv: {missing_customers}')
cost_dict = {}
for (_, row) in transportation_costs_df.iterrows():
    supplier = row['Unnamed: 0']
    for customer in customers:
        val = row[customer]
        try:
            cost = float(val)
        except Exception:
            raise ValueError(f"Invalid cost value for supplier '{supplier}', customer '{customer}': '{val}'")
        cost_dict[supplier, customer] = cost
for s in suppliers:
    if s not in supply_capacity_dict:
        raise ValueError(f"Supplier '{s}' missing in supply_capacity.csv")
for c in customers:
    if c not in demand_dict:
        raise ValueError(f"Customer '{c}' missing in customer_demand.csv")
    for s in suppliers:
        if (s, c) not in cost_dict:
            raise ValueError(f"Missing transportation cost for supplier '{s}', customer '{c}'")
m = gp.Model('TransportationProblem')
x_vars = m.addVars(suppliers, customers, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((cost_dict[i, j] * x_vars[i, j] for i in suppliers for j in customers)), gp.GRB.MINIMIZE)
for i in suppliers:
    m.addConstr(gp.quicksum((x_vars[i, j] for j in customers)) <= supply_capacity_dict[i], name=f'supply_capacity_{i}')
for j in customers:
    m.addConstr(gp.quicksum((x_vars[i, j] for i in suppliers)) == demand_dict[j], name=f'demand_{j}')
m.optimize()