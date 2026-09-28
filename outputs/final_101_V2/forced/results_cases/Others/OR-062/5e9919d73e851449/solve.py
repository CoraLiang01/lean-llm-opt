import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
demand_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP6/demand.csv', sep=',')
fixed_cost_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP6/fixed_cost.csv', sep=',')
trans_cost_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP6/transportation_costs.csv', sep=',')

def normalize_name(s):
    return re.sub('\\s+', ' ', str(s)).strip()
suppliers = [normalize_name(s) for s in fixed_cost_df['Unnamed: 0']]
suppliers_tc = [normalize_name(s) for s in trans_cost_df['Unnamed: 0']]
if set(suppliers) != set(suppliers_tc):
    raise ValueError(f'Supplier sets in fixed_cost.csv and transportation_costs.csv do not match: {suppliers} vs {suppliers_tc}')
supplier_idx_tc = {normalize_name(s): idx for idx, s in enumerate(trans_cost_df['Unnamed: 0'])}
customers = [normalize_name(c) for c in demand_df['Customer']]
customers_tc = [normalize_name(c) for c in trans_cost_df.columns if c != 'Unnamed: 0']
if set(customers) != set(customers_tc):
    raise ValueError(f'Customer sets in demand.csv and transportation_costs.csv do not match: {customers} vs {customers_tc}')
customer_idx_tc = {normalize_name(c): c for c in trans_cost_df.columns if c != 'Unnamed: 0'}
fixed_costs = {}
for idx, row in fixed_cost_df.iterrows():
    s = normalize_name(row['Unnamed: 0'])
    fixed_costs[s] = float(row['fixed_costs'])
demands = {}
for idx, row in demand_df.iterrows():
    c = normalize_name(row['Customer'])
    demands[c] = int(row['demand'])
transport_costs = {}
for s in suppliers:
    row_idx = supplier_idx_tc[s]
    row = trans_cost_df.iloc[row_idx]
    for c in customers:
        col = customer_idx_tc[c]
        cost = float(row[col])
        transport_costs[s, c] = cost
m = gp.Model('UFLP_Iowa_Liquor')
y = m.addVars(suppliers, vtype=gp.GRB.BINARY, name='')
x = m.addVars(suppliers, customers, vtype=gp.GRB.CONTINUOUS, lb=0.0, name='')
fixed_cost_term = gp.quicksum((fixed_costs[s] * y[s] for s in suppliers))
trans_cost_term = gp.quicksum((transport_costs[s, c] * x[s, c] for s in suppliers for c in customers))
m.setObjective(fixed_cost_term + trans_cost_term, gp.GRB.MINIMIZE)
for c in customers:
    m.addConstr(gp.quicksum((x[s, c] for s in suppliers)) == demands[c], name=f'demand_{c}')
M = sum(demands.values())
for s in suppliers:
    for c in customers:
        m.addConstr(x[s, c] <= M * y[s], name=f'link_{s}_{c}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f}\n')
    print('--- Supplier Activation ---')
    for s in suppliers:
        print(f"  Supplier '{s}': {('OPEN' if y[s].X > 0.5 else 'CLOSED')} (y={int(round(y[s].X))})")
    print('\n--- Shipment Plan (nonzero only) ---')
    for s in suppliers:
        for c in customers:
            if x[s, c].X > 1e-06:
                print(f"  {x[s, c].X:.2f} units from '{s}' to '{c}' (cost per unit: {transport_costs[s, c]:.2f})")
else:
    print(f'No optimal solution found. Status: {m.status}')