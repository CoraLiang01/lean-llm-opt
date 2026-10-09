import gurobipy as gp
import pandas as pd
import numpy as np
import re
customer_demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP6/customer_demand.csv'
supply_capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP6/supply_capacity.csv'
transportation_costs_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP6/transportation_costs.csv'
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
supply_capacity_df['warehouse'] = supply_capacity_df['Unnamed: 0'].str.strip()
supply_capacity_df['supply_capacity'] = supply_capacity_df['supply_capacity'].astype(float)
warehouses = supply_capacity_df['warehouse'].tolist()
supply_capacity_dict = dict(zip(supply_capacity_df['warehouse'], supply_capacity_df['supply_capacity']))
transportation_costs_df = pd.read_csv(transportation_costs_path, dtype=str, keep_default_na=False)
if 'Unnamed: 0' not in transportation_costs_df.columns:
    raise KeyError("transportation_costs.csv must contain column 'Unnamed: 0' for warehouse IDs")
transportation_costs_df['warehouse'] = transportation_costs_df['Unnamed: 0'].str.strip()
for c in customers:
    if c not in transportation_costs_df.columns:
        raise KeyError(f"transportation_costs.csv missing required customer column '{c}'")
cost_dict = {}
for (idx, row) in transportation_costs_df.iterrows():
    w = row['warehouse']
    for c in customers:
        val = row[c]
        try:
            cost = float(val)
        except Exception:
            raise ValueError(f"Invalid cost value for warehouse '{w}', customer '{c}': '{val}'")
        cost_dict[w, c] = cost
for w in warehouses:
    for c in customers:
        if (w, c) not in cost_dict:
            raise KeyError(f"Missing transportation cost for warehouse '{w}', customer '{c}'")
m = gp.Model('TransportationProblem')
x_vars = m.addVars(warehouses, customers, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((cost_dict[w, c] * x_vars[w, c] for w in warehouses for c in customers)), gp.GRB.MINIMIZE)
for c in customers:
    m.addConstr(gp.quicksum((x_vars[w, c] for w in warehouses)) == demand_dict[c], name=f'demand_{c}')
for w in warehouses:
    m.addConstr(gp.quicksum((x_vars[w, c] for c in customers)) <= supply_capacity_dict[w], name=f'supply_{w}')
m.optimize()