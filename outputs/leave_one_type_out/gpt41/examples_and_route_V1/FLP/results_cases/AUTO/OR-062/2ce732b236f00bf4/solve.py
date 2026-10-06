import gurobipy as gp
import pandas as pd
import numpy as np
import re
demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP6/demand.csv'
fixed_cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP6/fixed_cost.csv'
transport_cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP6/transportation_costs.csv'
demand_df = pd.read_csv(demand_path, sep=',')
customers = demand_df['Customer'].astype(str).tolist()
demand = dict(zip(demand_df['Customer'].astype(str), demand_df['demand']))
fixed_cost_df = pd.read_csv(fixed_cost_path, sep=',')

def normalize_name(s):
    return re.sub('\\s+', ' ', str(s)).strip()
fixed_cost_df['Supplier'] = fixed_cost_df['Unnamed: 0'].apply(normalize_name)
suppliers = fixed_cost_df['Supplier'].tolist()
fixed_cost = dict(zip(fixed_cost_df['Supplier'], fixed_cost_df['fixed_costs']))
transport_df = pd.read_csv(transport_cost_path, sep=',')
transport_df['Supplier'] = transport_df['Unnamed: 0'].apply(normalize_name)
transport_columns = [col for col in transport_df.columns if col not in ['Unnamed: 0', 'Supplier']]
customer_to_col = {}
for cust in customers:
    cust_norm = re.sub('[\\s_]+', '', cust).lower()
    found = None
    for col in transport_columns:
        col_norm = re.sub('[\\s_]+', '', col).lower()
        if cust_norm in col_norm or col_norm in cust_norm:
            found = col
            break
    if found is None:
        raise ValueError(f"Could not match customer '{cust}' to any transportation_costs.csv column.")
    customer_to_col[cust] = found
transport_cost = {}
for i, row in transport_df.iterrows():
    supplier = row['Supplier']
    for cust in customers:
        col = customer_to_col[cust]
        cost = row[col]
        transport_cost[supplier, cust] = float(cost)
for s in suppliers:
    if s not in fixed_cost:
        raise ValueError(f"Supplier '{s}' missing from fixed_costs.")
    for c in customers:
        if (s, c) not in transport_cost:
            raise ValueError(f"Missing transportation cost for supplier '{s}', customer '{c}'.")
m = gp.Model('UFLP_Iowa_Liquor')
y = m.addVars(suppliers, vtype=gp.GRB.BINARY, name='')
x = m.addVars(suppliers, customers, vtype=gp.GRB.CONTINUOUS, lb=0.0, name='')
fixed_cost_expr = gp.quicksum((fixed_cost[i] * y[i] for i in suppliers))
transport_cost_expr = gp.quicksum((transport_cost[i, j] * x[i, j] for i in suppliers for j in customers))
m.setObjective(fixed_cost_expr + transport_cost_expr, gp.GRB.MINIMIZE)
for j in customers:
    m.addConstr(gp.quicksum((x[i, j] for i in suppliers)) == demand[j], name=f'demand_{j}')
M = sum(demand.values())
for i in suppliers:
    for j in customers:
        m.addConstr(x[i, j] <= M * y[i], name=f'link_{i}_{j}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f}\n')
    print('--- Supplier Activation ---')
    for i in suppliers:
        print(f"  {i}: {('OPEN' if y[i].X > 0.5 else 'closed')} (y={int(round(y[i].X))})")
    print('\n--- Shipments (x[i,j]) ---')
    for i in suppliers:
        for j in customers:
            if x[i, j].X > 1e-06:
                print(f"  Supplier '{i}' -> Customer '{j}': {x[i, j].X:.2f}")
else:
    print(f'No optimal solution found. Status: {m.status}')