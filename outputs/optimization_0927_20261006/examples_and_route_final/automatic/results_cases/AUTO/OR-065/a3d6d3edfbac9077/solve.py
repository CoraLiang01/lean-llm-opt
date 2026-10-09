import gurobipy as gp
import pandas as pd
import numpy as np
import re
demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP9/demand.csv'
fixed_cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP9/fixed_cost.csv'
transport_cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP9/transportation_costs.csv'
demand_df = pd.read_csv(demand_path, dtype=str, keep_default_na=False)
fixed_cost_df = pd.read_csv(fixed_cost_path, dtype=str, keep_default_na=False)
transport_cost_df = pd.read_csv(transport_cost_path, dtype=str, keep_default_na=False)
customers = demand_df['customer'].str.strip().tolist()
warehouses_fc = fixed_cost_df['Unnamed: 0'].str.strip().tolist()
warehouses_tc = transport_cost_df['Unnamed: 0'].str.strip().tolist()
if set(warehouses_fc) != set(warehouses_tc):
    raise ValueError('Mismatch in warehouse identifiers between fixed_cost.csv and transportation_costs.csv')
warehouses = warehouses_fc
demand = {}
for (idx, row) in demand_df.iterrows():
    cust = row['customer'].strip()
    try:
        demand_val = int(row['demand'])
    except Exception:
        raise ValueError(f"Invalid demand value for customer {cust}: {row['demand']}")
    demand[cust] = demand_val
fixed_costs = {}
for (idx, row) in fixed_cost_df.iterrows():
    wh = row['Unnamed: 0'].strip()
    try:
        fc = float(row['fixed_costs'])
    except Exception:
        raise ValueError(f"Invalid fixed cost for warehouse {wh}: {row['fixed_costs']}")
    fixed_costs[wh] = fc
transport_costs = {}
for (idx, row) in transport_cost_df.iterrows():
    wh = row['Unnamed: 0'].strip()
    for cust in customers:
        if cust not in row:
            raise ValueError(f'Customer {cust} not found in transportation_costs.csv columns')
        try:
            tc = float(row[cust])
        except Exception:
            raise ValueError(f'Invalid transportation cost for warehouse {wh}, customer {cust}: {row[cust]}')
        transport_costs[wh, cust] = tc
if set(demand.keys()) != set(customers):
    raise ValueError('Mismatch in customer identifiers between demand.csv and code')
if set(fixed_costs.keys()) != set(warehouses):
    raise ValueError('Mismatch in warehouse identifiers between fixed_cost.csv and code')
for wh in warehouses:
    for cust in customers:
        if (wh, cust) not in transport_costs:
            raise ValueError(f'Missing transportation cost for warehouse {wh}, customer {cust}')
m = gp.Model('UFLP_Bandcamp')
y_vars = m.addVars(warehouses, vtype=gp.GRB.BINARY, name='')
x_vars = m.addVars(warehouses, customers, vtype=gp.GRB.CONTINUOUS, lb=0.0, name='')
m.setObjective(gp.quicksum((fixed_costs[wh] * y_vars[wh] for wh in warehouses)) + gp.quicksum((transport_costs[wh, cust] * x_vars[wh, cust] for wh in warehouses for cust in customers)), gp.GRB.MINIMIZE)
for cust in customers:
    m.addConstr(gp.quicksum((x_vars[wh, cust] for wh in warehouses)) == demand[cust], name=f'demand_{cust}')
for wh in warehouses:
    for cust in customers:
        m.addConstr(x_vars[wh, cust] <= demand[cust] * y_vars[wh], name=f'link_{wh}_{cust}')
m.optimize()