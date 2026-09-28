import gurobipy as gp
import pandas as pd
import numpy as np
demand_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP2/demand.csv', sep=',')
demand_df['customer'] = demand_df['customer'].astype(str).str.strip()
customers = demand_df['customer'].tolist()
demand = dict(zip(demand_df['customer'], demand_df['demand']))
fixed_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP2/fixed_cost.csv', sep=',')
fixed_df['supplier'] = fixed_df['Unnamed: 0'].astype(str).str.strip()
suppliers = fixed_df['supplier'].tolist()
fixed_costs = dict(zip(fixed_df['supplier'], fixed_df['fixed_costs']))
trans_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP2/transportation_costs.csv', sep=',')
trans_df['supplier'] = trans_df['Unnamed: 0'].astype(str).str.strip()
if set(suppliers) != set(trans_df['supplier']):
    raise ValueError('Mismatch between suppliers in fixed_cost.csv and transportation_costs.csv')
if set(customers) - set(trans_df.columns[1:]):
    raise ValueError('Some customers in demand.csv are missing in transportation_costs.csv')
transportation_costs = {}
for _, row in trans_df.iterrows():
    i = row['supplier']
    for j in customers:
        transportation_costs[i, j] = float(row[j])
m = gp.Model('UFLP_Colorado_Motor_Vehicle_Sales')
x = m.addVars(suppliers, customers, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
y = m.addVars(suppliers, vtype=gp.GRB.BINARY, name='')
obj = gp.quicksum((fixed_costs[i] * y[i] for i in suppliers)) + gp.quicksum((transportation_costs[i, j] * x[i, j] for i in suppliers for j in customers))
m.setObjective(obj, gp.GRB.MINIMIZE)
for j in customers:
    m.addConstr(gp.quicksum((x[i, j] for i in suppliers)) == demand[j], name=f'demand_{j}')
M = sum(demand.values())
for i in suppliers:
    for j in customers:
        m.addConstr(x[i, j] <= M * y[i], name=f'link_{i}_{j}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f}')
    print('\n--- Supplier Activation ---')
    for i in suppliers:
        print(f"  Supplier {i}: {('OPEN' if y[i].X > 0.5 else 'CLOSED')} (y={int(round(y[i].X))})")
    print('\n--- Supply Plan (vehicles from supplier to dealership) ---')
    for i in suppliers:
        for j in customers:
            if x[i, j].X > 0.001:
                print(f'  {x[i, j].X:.0f} vehicles from {i} to {j} (cost per vehicle: {transportation_costs[i, j]})')
else:
    print(f'No optimal solution found. Status: {m.status}')