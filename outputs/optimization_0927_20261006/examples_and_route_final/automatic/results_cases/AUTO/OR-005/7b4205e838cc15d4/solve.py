import gurobipy as gp
import pandas as pd
import numpy as np
import re
customer_demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP5/customer_demand.csv'
supply_capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP5/supply_capacity.csv'
transportation_costs_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP5/transportation_costs.csv'
customer_demand_df = pd.read_csv(customer_demand_path, dtype=str, keep_default_na=False)
customer_demand_df['Customers'] = customer_demand_df['Customers'].str.strip()
customer_demand_df['demand'] = customer_demand_df['demand'].astype(float)
customers = list(customer_demand_df['Customers'])
supply_capacity_df = pd.read_csv(supply_capacity_path, dtype=str, keep_default_na=False)
supply_capacity_df['Supplier'] = supply_capacity_df['Supplier'].str.strip()
supply_capacity_df['supply_capacity'] = supply_capacity_df['supply_capacity'].astype(float)
suppliers = list(supply_capacity_df['Supplier'])
transportation_costs_df = pd.read_csv(transportation_costs_path, dtype=str, keep_default_na=False)
transportation_costs_df['Unnamed: 0'] = transportation_costs_df['Unnamed: 0'].str.strip()
cost_suppliers = list(transportation_costs_df['Unnamed: 0'])
cost_customers = [col for col in transportation_costs_df.columns if col != 'Unnamed: 0']
if set(suppliers) != set(cost_suppliers):
    raise ValueError(f'Supplier mismatch between supply_capacity.csv and transportation_costs.csv: {set(suppliers)} vs {set(cost_suppliers)}')
if set(customers) != set(cost_customers):
    raise ValueError(f'Customer mismatch between customer_demand.csv and transportation_costs.csv: {set(customers)} vs {set(cost_customers)}')
cost = {}
for (_, row) in transportation_costs_df.iterrows():
    s = row['Unnamed: 0']
    for c in customers:
        val = row[c]
        try:
            cost[s, c] = float(val)
        except Exception:
            raise ValueError(f"Invalid cost value for supplier '{s}', customer '{c}': {val}")
demand = dict(zip(customer_demand_df['Customers'], customer_demand_df['demand']))
supply_capacity = dict(zip(supply_capacity_df['Supplier'], supply_capacity_df['supply_capacity']))
m = gp.Model('TransportationProblem')
x_vars = m.addVars(suppliers, customers, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((cost[s, c] * x_vars[s, c] for s in suppliers for c in customers)), gp.GRB.MINIMIZE)
for c in customers:
    m.addConstr(gp.quicksum((x_vars[s, c] for s in suppliers)) == demand[c], name=f'demand_{c}')
for s in suppliers:
    m.addConstr(gp.quicksum((x_vars[s, c] for c in customers)) <= supply_capacity[s], name=f'supply_{s}')
m.optimize()