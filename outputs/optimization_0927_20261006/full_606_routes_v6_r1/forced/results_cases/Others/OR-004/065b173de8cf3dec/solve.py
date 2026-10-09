import gurobipy as gp
import pandas as pd
import numpy as np
customer_demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP4/customer_demand.csv'
supply_capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP4/supply_capacity.csv'
transportation_costs_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP4/transportation_costs.csv'
customer_demand_df = pd.read_csv(customer_demand_path, dtype=str, keep_default_na=False)
if 'customer' not in customer_demand_df.columns or 'demand' not in customer_demand_df.columns:
    raise KeyError("customer_demand.csv must contain columns 'customer' and 'demand'")
customer_demand_df['customer'] = customer_demand_df['customer'].str.strip()
customer_demand_df['demand'] = customer_demand_df['demand'].astype(float)
customers = customer_demand_df['customer'].tolist()
demand_dict = dict(zip(customer_demand_df['customer'], customer_demand_df['demand']))
supply_capacity_df = pd.read_csv(supply_capacity_path, dtype=str, keep_default_na=False)
if 'Unnamed: 0' not in supply_capacity_df.columns or 'supply_capacity' not in supply_capacity_df.columns:
    raise KeyError("supply_capacity.csv must contain columns 'Unnamed: 0' and 'supply_capacity'")
supply_capacity_df['supply'] = supply_capacity_df['Unnamed: 0'].str.strip()
supply_capacity_df['supply_capacity'] = supply_capacity_df['supply_capacity'].astype(float)
supplies = supply_capacity_df['supply'].tolist()
supply_capacity_dict = dict(zip(supply_capacity_df['supply'], supply_capacity_df['supply_capacity']))
transportation_costs_df = pd.read_csv(transportation_costs_path, dtype=str, keep_default_na=False)
if 'Unnamed: 0' not in transportation_costs_df.columns:
    raise KeyError("transportation_costs.csv must contain column 'Unnamed: 0' for supply IDs")
transportation_costs_df['supply'] = transportation_costs_df['Unnamed: 0'].str.strip()
for c in customers:
    if c not in transportation_costs_df.columns:
        raise KeyError(f"transportation_costs.csv missing required customer column '{c}'")
for c in customers:
    transportation_costs_df[c] = transportation_costs_df[c].astype(float)
cost_dict = {}
for (_, row) in transportation_costs_df.iterrows():
    s = row['supply']
    for c in customers:
        cost_dict[s, c] = float(row[c])
if set(supplies) != set(transportation_costs_df['supply']):
    raise ValueError('Mismatch between supplies in supply_capacity.csv and transportation_costs.csv')
if set(customers) != set(transportation_costs_df.columns) - {'Unnamed: 0', 'supply'}:
    raise ValueError('Mismatch between customers in customer_demand.csv and transportation_costs.csv')
m = gp.Model('TransportationOptimization')
x_vars = m.addVars(supplies, customers, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((cost_dict[s, c] * x_vars[s, c] for s in supplies for c in customers)), gp.GRB.MINIMIZE)
for s in supplies:
    m.addConstr(gp.quicksum((x_vars[s, c] for c in customers)) <= supply_capacity_dict[s], name=f'supply_capacity_{s}')
for c in customers:
    m.addConstr(gp.quicksum((x_vars[s, c] for s in supplies)) == demand_dict[c], name=f'demand_{c}')
m.optimize()