import gurobipy as gp
import pandas as pd
import numpy as np
import re
customer_demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP7/customer_demand.csv'
supply_capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP7/supply_capacity.csv'
transportation_costs_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP7/transportation_costs.csv'
customer_demand_df = pd.read_csv(customer_demand_path, dtype=str, keep_default_na=False)
if not {'customer', 'demand'}.issubset(customer_demand_df.columns):
    raise ValueError('customer_demand.csv missing required columns.')
customer_demand_df['customer'] = customer_demand_df['customer'].str.strip()
customer_demand_df['demand'] = customer_demand_df['demand'].astype(float)
store_ids = customer_demand_df['customer'].tolist()
store_demand = dict(zip(customer_demand_df['customer'], customer_demand_df['demand']))
supply_capacity_df = pd.read_csv(supply_capacity_path, dtype=str, keep_default_na=False)
if not {'region', 'supply_capacity'}.issubset(supply_capacity_df.columns):
    raise ValueError('supply_capacity.csv missing required columns.')
supply_capacity_df['region'] = supply_capacity_df['region'].str.strip()
supply_capacity_df['supply_capacity'] = supply_capacity_df['supply_capacity'].astype(float)
warehouse_ids = supply_capacity_df['region'].tolist()
warehouse_capacity = dict(zip(supply_capacity_df['region'], supply_capacity_df['supply_capacity']))
transportation_costs_df = pd.read_csv(transportation_costs_path, dtype=str, keep_default_na=False)
if 'Unnamed: 0' not in transportation_costs_df.columns:
    raise ValueError("transportation_costs.csv missing 'Unnamed: 0' column for warehouse IDs.")
transportation_costs_df['Unnamed: 0'] = transportation_costs_df['Unnamed: 0'].str.strip()
cost_warehouse_ids = transportation_costs_df['Unnamed: 0'].tolist()
cost_store_ids = [col for col in transportation_costs_df.columns if col != 'Unnamed: 0']
if set(warehouse_ids) != set(cost_warehouse_ids):
    raise ValueError(f'Warehouse IDs in supply_capacity.csv and transportation_costs.csv do not match: {warehouse_ids} vs {cost_warehouse_ids}')
if set(store_ids) != set(cost_store_ids):
    raise ValueError(f'Store IDs in customer_demand.csv and transportation_costs.csv do not match: {store_ids} vs {cost_store_ids}')
transportation_costs = {}
for (_, row) in transportation_costs_df.iterrows():
    w = row['Unnamed: 0']
    for s in store_ids:
        val = row[s]
        try:
            cost = float(val)
        except Exception:
            raise ValueError(f'Invalid cost value for warehouse {w}, store {s}: {val}')
        transportation_costs[w, s] = cost
decision_keys = [(w, s) for w in warehouse_ids for s in store_ids]
for key in decision_keys:
    if key not in transportation_costs:
        raise ValueError(f'Missing transportation cost for warehouse {key[0]}, store {key[1]}')

def solve_greenmart_transportation():
    m = gp.Model('GreenMart_Transportation')
    quantity_vars = m.addVars(decision_keys, lb=0, vtype=gp.GRB.CONTINUOUS, name='')
    m.setObjective(gp.quicksum((transportation_costs[w, s] * quantity_vars[w, s] for (w, s) in decision_keys)), gp.GRB.MINIMIZE)
    for s in store_ids:
        m.addConstr(gp.quicksum((quantity_vars[w, s] for w in warehouse_ids)) == store_demand[s], name=f'demand_{s}')
    for w in warehouse_ids:
        m.addConstr(gp.quicksum((quantity_vars[w, s] for s in store_ids)) <= warehouse_capacity[w], name=f'supply_{w}')
    m.Params.MIPGap = 0.0001
    m.optimize()
    return m
m = solve_greenmart_transportation()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName} {var.X}')
else:
    print(f'Solver status: {m.status}')