import gurobipy as gp
import pandas as pd
import numpy as np
import math
demand_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP6/demand.csv', sep=',')
customers = demand_df['Customer'].astype(str).tolist()
demand = dict(zip(demand_df['Customer'].astype(str), demand_df['demand']))
fixed_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP6/fixed_cost.csv', sep=',')
suppliers = fixed_df['Unnamed: 0'].astype(str).str.strip().tolist()
fixed_costs = dict(zip(fixed_df['Unnamed: 0'].astype(str).str.strip(), fixed_df['fixed_costs']))
trans_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP6/transportation_costs.csv', sep=',')
trans_df['Supplier'] = trans_df['Unnamed: 0'].astype(str).str.strip()
store_cols = [col for col in trans_df.columns if col not in ['Unnamed: 0', 'Supplier']]
if set(store_cols) != set(customers):
    raise ValueError(f'Mismatch between store columns in transportation_costs.csv and customers in demand.csv.\nStores: {store_cols}\nCustomers: {customers}')
transportation_cost = {}
for (_, row) in trans_df.iterrows():
    supplier = str(row['Supplier'])
    for store in store_cols:
        customer = str(store)
        transportation_cost[supplier, customer] = float(row[store])
if set(suppliers) != set(trans_df['Supplier']):
    raise ValueError('Mismatch between suppliers in fixed_cost.csv and transportation_costs.csv.')
m = gp.Model('UFLP_Iowa_Liquor')
x = m.addVars(suppliers, customers, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
y = m.addVars(suppliers, vtype=gp.GRB.BINARY, name='')
total_fixed = gp.quicksum((fixed_costs[i] * y[i] for i in suppliers))
total_trans = gp.quicksum((transportation_cost[i, j] * x[i, j] for i in suppliers for j in customers))
m.setObjective(total_fixed + total_trans, gp.GRB.MINIMIZE)
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
        print(f"  Supplier '{i}': {('OPEN' if y[i].X > 0.5 else 'CLOSED')} (y={int(round(y[i].X))})")
    print('\n--- Shipment Plan (x[i,j]) ---')
    for i in suppliers:
        for j in customers:
            if x[i, j].X > 1e-06:
                print(f"  {x[i, j].X:.2f} units from Supplier '{i}' to Customer '{j}' (cost per unit: {transportation_cost[i, j]})")
else:
    print(f'No optimal solution found. Status: {m.status}')