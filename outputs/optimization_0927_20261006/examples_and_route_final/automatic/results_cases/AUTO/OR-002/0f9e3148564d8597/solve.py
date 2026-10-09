import gurobipy as gp
import pandas as pd
import numpy as np
import re
customer_demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP2/customer_demand.csv'
supply_capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP2/supply_capacity.csv'
transportation_costs_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP2/transportation_costs.csv'
customer_demand_df = pd.read_csv(customer_demand_path, dtype=str, keep_default_na=False)
supply_capacity_df = pd.read_csv(supply_capacity_path, dtype=str, keep_default_na=False)
transportation_costs_df = pd.read_csv(transportation_costs_path, dtype=str, keep_default_na=False)
customers = customer_demand_df['customer'].astype(str).tolist()
demand = {}
for (idx, row) in customer_demand_df.iterrows():
    cust = str(row['customer'])
    try:
        demand_val = int(row['demand'])
    except Exception:
        raise ValueError(f"Invalid demand value for customer {cust}: {row['demand']}")
    demand[cust] = demand_val
stores = supply_capacity_df['Unnamed: 0'].astype(str).tolist()
supply_capacity = {}
for (idx, row) in supply_capacity_df.iterrows():
    store = str(row['Unnamed: 0'])
    try:
        cap_val = int(row['supply_capacity'])
    except Exception:
        raise ValueError(f"Invalid supply_capacity value for store {store}: {row['supply_capacity']}")
    supply_capacity[store] = cap_val
cost = {}
cost_customer_cols = [col for col in transportation_costs_df.columns if col != 'Unnamed: 0']
missing_customers = set(customers) - set(cost_customer_cols)
if missing_customers:
    raise ValueError(f'Missing customer columns in transportation_costs.csv: {missing_customers}')
for (idx, row) in transportation_costs_df.iterrows():
    store = str(row['Unnamed: 0'])
    if store not in stores:
        continue
    cost[store] = {}
    for cust in customers:
        val = row[cust]
        try:
            cost_val = float(val)
        except Exception:
            raise ValueError(f'Invalid cost value for store {store}, customer {cust}: {val}')
        cost[store][cust] = cost_val
for s in stores:
    if s not in cost:
        raise ValueError(f'Store {s} missing in transportation_costs.csv')
    for c in customers:
        if c not in cost[s]:
            raise ValueError(f'Cost for store {s}, customer {c} missing in transportation_costs.csv')
m = gp.Model('Walmart_Transportation')
x_vars = m.addVars(stores, customers, vtype=gp.GRB.CONTINUOUS, lb=0.0, name='')
m.setObjective(gp.quicksum((cost[s][c] * x_vars[s, c] for s in stores for c in customers)), gp.GRB.MINIMIZE)
for c in customers:
    m.addConstr(gp.quicksum((x_vars[s, c] for s in stores)) == demand[c], name='')
for s in stores:
    m.addConstr(gp.quicksum((x_vars[s, c] for c in customers)) <= supply_capacity[s], name='')
m.optimize()