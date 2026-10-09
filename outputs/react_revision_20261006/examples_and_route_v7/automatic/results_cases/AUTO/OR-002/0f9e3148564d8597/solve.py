import gurobipy as gp
import pandas as pd
import numpy as np
customer_demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP2/customer_demand.csv'
supply_capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP2/supply_capacity.csv'
transportation_costs_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP2/transportation_costs.csv'
customer_df = pd.read_csv(customer_demand_path, dtype=str, keep_default_na=False)
if not {'customer', 'demand'}.issubset(customer_df.columns):
    raise ValueError("customer_demand.csv must contain columns 'customer' and 'demand'")
customer_df['customer'] = customer_df['customer'].str.strip()
customer_df['demand'] = customer_df['demand'].astype(float)
customers = customer_df['customer'].tolist()
customer_demand = dict(zip(customer_df['customer'], customer_df['demand']))
supply_df = pd.read_csv(supply_capacity_path, dtype=str, keep_default_na=False)
if not {'Unnamed: 0', 'supply_capacity'}.issubset(supply_df.columns):
    raise ValueError("supply_capacity.csv must contain columns 'Unnamed: 0' and 'supply_capacity'")
supply_df['store'] = supply_df['Unnamed: 0'].str.strip()
supply_df['supply_capacity'] = supply_df['supply_capacity'].astype(float)
stores = supply_df['store'].tolist()
supply_capacity = dict(zip(supply_df['store'], supply_df['supply_capacity']))
costs_df = pd.read_csv(transportation_costs_path, dtype=str, keep_default_na=False)
if 'Unnamed: 0' not in costs_df.columns:
    raise ValueError("transportation_costs.csv must contain column 'Unnamed: 0' for store IDs")
costs_df['store'] = costs_df['Unnamed: 0'].str.strip()
missing_customers = [c for c in customers if c not in costs_df.columns]
if missing_customers:
    raise ValueError(f'transportation_costs.csv missing columns for customers: {missing_customers}')
transportation_cost = {}
for (_, row) in costs_df.iterrows():
    s = row['store']
    for c in customers:
        val = row[c]
        try:
            cost = float(val)
        except Exception:
            raise ValueError(f'Invalid cost value for store {s}, customer {c}: {val}')
        transportation_cost[s, c] = cost
cost_stores = [str(x).strip() for x in costs_df['store']]
if set(stores) != set(cost_stores):
    raise ValueError(f'Stores in supply_capacity.csv and transportation_costs.csv do not match: {set(stores) ^ set(cost_stores)}')
cost_customers = [c for c in customers]
if not all((c in costs_df.columns for c in cost_customers)):
    raise ValueError('Mismatch between customers in customer_demand.csv and columns in transportation_costs.csv')
m = gp.Model('Walmart_Transportation')
quantity_keys = [(s, c) for s in stores for c in customers]
quantity_vars = m.addVars(quantity_keys, lb=0, vtype=gp.GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((transportation_cost[s, c] * quantity_vars[s, c] for (s, c) in quantity_keys)), gp.GRB.MINIMIZE)
for c in customers:
    m.addConstr(gp.quicksum((quantity_vars[s, c] for s in stores)) == customer_demand[c], name=f'demand_{c}')
for s in stores:
    m.addConstr(gp.quicksum((quantity_vars[s, c] for c in customers)) <= supply_capacity[s], name=f'supply_{s}')
m.Params.MIPGap = 0.0001
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for (s, c) in quantity_keys:
        var = quantity_vars[s, c]
        print(f'{var.VarName} {var.X}')
else:
    print(f'Solver status: {m.status}')