import gurobipy as gp
import pandas as pd
import numpy as np
import re
customer_demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP4/customer_demand.csv'
supply_capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP4/supply_capacity.csv'
transportation_costs_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP4/transportation_costs.csv'
customer_demand_df = pd.read_csv(customer_demand_path, dtype=str, keep_default_na=False)
if 'customer' not in customer_demand_df.columns or 'demand' not in customer_demand_df.columns:
    raise ValueError("customer_demand.csv must contain columns 'customer' and 'demand'")
customer_demand_df['customer'] = customer_demand_df['customer'].astype(str).str.strip()
customer_demand_df['demand'] = customer_demand_df['demand'].astype(float)
customers = customer_demand_df['customer'].tolist()
demand = dict(zip(customer_demand_df['customer'], customer_demand_df['demand']))
supply_capacity_df = pd.read_csv(supply_capacity_path, dtype=str, keep_default_na=False)
if 'Unnamed: 0' not in supply_capacity_df.columns or 'supply_capacity' not in supply_capacity_df.columns:
    raise ValueError("supply_capacity.csv must contain columns 'Unnamed: 0' and 'supply_capacity'")
supply_capacity_df['source'] = supply_capacity_df['Unnamed: 0'].astype(str).str.strip()
supply_capacity_df['supply_capacity'] = supply_capacity_df['supply_capacity'].astype(float)
sources = supply_capacity_df['source'].tolist()
supply_capacity = dict(zip(supply_capacity_df['source'], supply_capacity_df['supply_capacity']))
transportation_costs_df = pd.read_csv(transportation_costs_path, dtype=str, keep_default_na=False)
if 'Unnamed: 0' not in transportation_costs_df.columns:
    raise ValueError("transportation_costs.csv must contain column 'Unnamed: 0' for sources")
transportation_costs_df['source'] = transportation_costs_df['Unnamed: 0'].astype(str).str.strip()
cost_customer_cols = [col for col in transportation_costs_df.columns if re.fullmatch('C\\d+', col)]
if set(customers) != set(cost_customer_cols):
    raise ValueError(f'Mismatch between customers in customer_demand.csv and columns in transportation_costs.csv.\ncustomer_demand.csv: {sorted(customers)}\ntransportation_costs.csv: {sorted(cost_customer_cols)}')
if set(sources) != set(transportation_costs_df['source']):
    raise ValueError(f"Mismatch between sources in supply_capacity.csv and rows in transportation_costs.csv.\nsupply_capacity.csv: {sorted(sources)}\ntransportation_costs.csv: {sorted(transportation_costs_df['source'])}")
cost = {}
for (_, row) in transportation_costs_df.iterrows():
    s = row['source']
    for c in customers:
        try:
            cost_val = float(row[c])
        except Exception as e:
            raise ValueError(f'Invalid cost value for source {s}, customer {c}: {row[c]}')
        cost[s, c] = cost_val
m = gp.Model('TransportationOptimization')
x_vars = m.addVars(sources, customers, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((cost[s, c] * x_vars[s, c] for s in sources for c in customers)), sense=gp.GRB.MINIMIZE)
for s in sources:
    m.addConstr(gp.quicksum((x_vars[s, c] for c in customers)) <= supply_capacity[s], name=f'supply_{s}')
for c in customers:
    m.addConstr(gp.quicksum((x_vars[s, c] for s in sources)) == demand[c], name=f'demand_{c}')
m.optimize()