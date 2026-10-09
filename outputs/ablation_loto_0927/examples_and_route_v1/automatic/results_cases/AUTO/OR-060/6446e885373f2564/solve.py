import gurobipy as gp
import pandas as pd
import numpy as np
import re
demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP3/demand.csv'
fixed_cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP3/fixed_cost.csv'
transport_cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP3/transportation_costs.csv'
demand_df = pd.read_csv(demand_path, sep=',')
demand_df['customer'] = demand_df['customer'].astype(str).str.strip()
customers = list(demand_df['customer'])
demand_dict = dict(zip(demand_df['customer'], demand_df['demand']))
fixed_cost_df = pd.read_csv(fixed_cost_path, sep=',')
fixed_cost_df['Unnamed: 0'] = fixed_cost_df['Unnamed: 0'].astype(str).str.strip()
suppliers = list(fixed_cost_df['Unnamed: 0'])
fixed_cost_dict = dict(zip(fixed_cost_df['Unnamed: 0'], fixed_cost_df['fixed_costs']))
trans_cost_df = pd.read_csv(transport_cost_path, sep=',')
trans_cost_df['Unnamed: 0'] = trans_cost_df['Unnamed: 0'].astype(str).str.strip()
for c in customers:
    if c not in trans_cost_df.columns:
        raise KeyError(f"Customer '{c}' not found in transportation_costs.csv columns.")
transport_cost = {}
for (idx, row) in trans_cost_df.iterrows():
    supplier = str(row['Unnamed: 0']).strip()
    if supplier not in suppliers:
        continue
    for customer in customers:
        cost = row[customer]
        if pd.isnull(cost):
            raise ValueError(f"Missing transportation cost for supplier '{supplier}', customer '{customer}'.")
        transport_cost[supplier, customer] = float(cost)
if set(suppliers) != set(trans_cost_df['Unnamed: 0'].unique()):
    missing = set(suppliers) - set(trans_cost_df['Unnamed: 0'].unique())
    if missing:
        raise ValueError(f'Suppliers missing in transportation_costs.csv: {missing}')
if set(customers) != set(demand_df['customer'].unique()):
    missing = set(customers) - set(demand_df['customer'].unique())
    if missing:
        raise ValueError(f'Customers missing in demand.csv: {missing}')
m = gp.Model('UFLP')
y = m.addVars(suppliers, vtype=gp.GRB.BINARY, name='')
x = m.addVars(suppliers, customers, vtype=gp.GRB.CONTINUOUS, lb=0.0, name='')
fixed_cost_term = gp.quicksum((fixed_cost_dict[i] * y[i] for i in suppliers))
transport_cost_term = gp.quicksum((transport_cost[i, j] * x[i, j] for i in suppliers for j in customers))
m.setObjective(fixed_cost_term + transport_cost_term, gp.GRB.MINIMIZE)
for j in customers:
    m.addConstr(gp.quicksum((x[i, j] for i in suppliers)) == demand_dict[j], name=f'demand_{j}')
for i in suppliers:
    for j in customers:
        m.addConstr(x[i, j] <= demand_dict[j] * y[i], name=f'suppact_{i}_{j}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f}\n')
    print('--- Supplier Opening Decisions ---')
    for i in suppliers:
        print(f"  Supplier {i}: {('OPEN' if y[i].X > 0.5 else 'closed')} (y={int(round(y[i].X))})")
    print('\n--- Supply Plan (x[i,j] > 0) ---')
    for i in suppliers:
        for j in customers:
            if x[i, j].X > 1e-06:
                print(f'  Supplier {i} -> Customer {j}: {x[i, j].X:.2f}')
else:
    print(f'No optimal solution found. Status: {m.status}')