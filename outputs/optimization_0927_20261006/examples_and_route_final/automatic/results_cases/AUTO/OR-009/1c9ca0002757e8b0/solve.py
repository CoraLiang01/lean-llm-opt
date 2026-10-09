import gurobipy as gp
import pandas as pd
import numpy as np
import re
customer_demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP9/customer_demand.csv'
supply_capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP9/supply_capacity.csv'
transportation_costs_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP9/transportation_costs.csv'
customer_demand_df = pd.read_csv(customer_demand_path, dtype=str, keep_default_na=False)
if not {'customer', 'demand'}.issubset(customer_demand_df.columns):
    raise ValueError("customer_demand.csv must contain columns: 'customer', 'demand'")
customer_demand_df['customer'] = customer_demand_df['customer'].str.strip()
customer_demand_df['demand'] = customer_demand_df['demand'].astype(float)
customers = customer_demand_df['customer'].tolist()
customer_demand = dict(zip(customer_demand_df['customer'], customer_demand_df['demand']))
supply_capacity_df = pd.read_csv(supply_capacity_path, dtype=str, keep_default_na=False)
if not {'Unnamed: 0', 'supply_capacity'}.issubset(supply_capacity_df.columns):
    raise ValueError("supply_capacity.csv must contain columns: 'Unnamed: 0', 'supply_capacity'")
supply_capacity_df['plant'] = supply_capacity_df['Unnamed: 0'].str.strip()
supply_capacity_df['supply_capacity'] = supply_capacity_df['supply_capacity'].astype(float)
plants = supply_capacity_df['plant'].tolist()
plant_capacity = dict(zip(supply_capacity_df['plant'], supply_capacity_df['supply_capacity']))
transportation_costs_df = pd.read_csv(transportation_costs_path, dtype=str, keep_default_na=False)
if 'Unnamed: 0' not in transportation_costs_df.columns:
    raise ValueError("transportation_costs.csv must contain column: 'Unnamed: 0'")
transportation_costs_df['plant'] = transportation_costs_df['Unnamed: 0'].str.strip()
cost_customer_cols = [col for col in transportation_costs_df.columns if col in customers]
if set(cost_customer_cols) != set(customers):
    raise ValueError(f'Mismatch between customers in demand and transportation_costs.csv: {set(customers)} vs {set(cost_customer_cols)}')
cost_plant_rows = transportation_costs_df['plant'].tolist()
if set(cost_plant_rows) != set(plants):
    raise ValueError(f'Mismatch between plants in supply_capacity and transportation_costs.csv: {set(plants)} vs {set(cost_plant_rows)}')
cost = {}
for (_, row) in transportation_costs_df.iterrows():
    plant = row['plant']
    for customer in customers:
        val = row[customer]
        try:
            cost_val = float(val)
        except Exception:
            raise ValueError(f'Invalid cost value for plant {plant}, customer {customer}: {val}')
        cost[plant, customer] = cost_val
m = gp.Model('BrewCo_Transportation')
x_vars = m.addVars(plants, customers, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((cost[s, c] * x_vars[s, c] for s in plants for c in customers)), gp.GRB.MINIMIZE)
for c in customers:
    m.addConstr(gp.quicksum((x_vars[s, c] for s in plants)) == customer_demand[c], name=f'demand_{c}')
for s in plants:
    m.addConstr(gp.quicksum((x_vars[s, c] for c in customers)) <= plant_capacity[s], name=f'supply_{s}')
m.optimize()