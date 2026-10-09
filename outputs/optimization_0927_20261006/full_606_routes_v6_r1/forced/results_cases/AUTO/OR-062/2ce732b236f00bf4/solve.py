import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP6/demand.csv'
fixed_cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP6/fixed_cost.csv'
transport_cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP6/transportation_costs.csv'
demand_df = pd.read_csv(demand_path, dtype=str, keep_default_na=False)
fixed_cost_df = pd.read_csv(fixed_cost_path, dtype=str, keep_default_na=False)
transport_df = pd.read_csv(transport_cost_path, dtype=str, keep_default_na=False)
customer_ids = demand_df['Customer'].tolist()
demand_dict = {}
for (idx, row) in demand_df.iterrows():
    cust = row['Customer']
    try:
        demand_val = int(row['demand'])
    except Exception as e:
        raise ValueError(f"Invalid demand value for customer {cust}: {row['demand']}")
    demand_dict[cust] = demand_val

def normalize_id(s):
    return s.strip()
supplier_ids = [normalize_id(s) for s in fixed_cost_df['Unnamed: 0'].tolist()]
transport_supplier_ids = [normalize_id(s) for s in transport_df['Unnamed: 0'].tolist()]
if set(supplier_ids) != set(transport_supplier_ids):
    raise ValueError(f'Supplier IDs in fixed_cost.csv and transportation_costs.csv do not match:\n{supplier_ids}\n{transport_supplier_ids}')
fixed_cost_dict = {}
for (idx, row) in fixed_cost_df.iterrows():
    sid = normalize_id(row['Unnamed: 0'])
    try:
        fc = float(row['fixed_costs'])
    except Exception as e:
        raise ValueError(f"Invalid fixed_costs value for supplier {sid}: {row['fixed_costs']}")
    fixed_cost_dict[sid] = fc
transport_store_cols = [c for c in transport_df.columns if c != 'Unnamed: 0']
if set(customer_ids) != set(transport_store_cols):
    raise ValueError(f'Customer IDs in demand.csv and transportation_costs.csv columns do not match:\n{customer_ids}\n{transport_store_cols}')
store_ids = customer_ids
transport_cost_dict = {}
for (idx, row) in transport_df.iterrows():
    sid = normalize_id(row['Unnamed: 0'])
    for store in store_ids:
        try:
            tc = float(row[store])
        except Exception as e:
            raise ValueError(f'Invalid transportation cost for supplier {sid}, store {store}: {row[store]}')
        transport_cost_dict[sid, store] = tc
suppliers = supplier_ids
stores = store_ids
m = gp.Model('Iowa_Liquor_UFLP')
x_vars = m.addVars(suppliers, stores, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
y_vars = m.addVars(suppliers, vtype=gp.GRB.BINARY, name='')
total_fixed_cost = gp.quicksum((fixed_cost_dict[i] * y_vars[i] for i in suppliers))
total_transport_cost = gp.quicksum((transport_cost_dict[i, j] * x_vars[i, j] for i in suppliers for j in stores))
m.setObjective(total_fixed_cost + total_transport_cost, gp.GRB.MINIMIZE)
for j in stores:
    m.addConstr(gp.quicksum((x_vars[i, j] for i in suppliers)) == demand_dict[j])
M = sum(demand_dict.values())
for i in suppliers:
    for j in stores:
        m.addConstr(x_vars[i, j] <= M * y_vars[i])
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f}')
    print('\n--- Supplier Activation ---')
    for i in suppliers:
        print(f"  Supplier '{i}': {('OPEN' if y_vars[i].X > 0.5 else 'CLOSED')} (y={int(round(y_vars[i].X))})")
    print('\n--- Shipment Plan (x[i,j]) ---')
    for i in suppliers:
        for j in stores:
            if x_vars[i, j].X > 1e-06:
                print(f"  Supplier '{i}' -> Store '{j}': {x_vars[i, j].X:.2f}")
else:
    print(f'No optimal solution found. Status: {m.status}')