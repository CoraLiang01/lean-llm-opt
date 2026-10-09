import gurobipy as gp
import pandas as pd
import numpy as np
demand_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP2/demand.csv', sep=',')
dealerships = demand_df['customer'].astype(str).tolist()
demand = demand_df.set_index('customer')['demand'].astype(float).to_dict()
fixed_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP2/fixed_cost.csv', sep=',')
suppliers = fixed_df['Unnamed: 0'].astype(str).tolist()
fixed_costs = fixed_df.set_index('Unnamed: 0')['fixed_costs'].astype(float).to_dict()
trans_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP2/transportation_costs.csv', sep=',')
trans_df['Unnamed: 0'] = trans_df['Unnamed: 0'].astype(str)
trans_df = trans_df.set_index('Unnamed: 0')
for c in dealerships:
    if c not in trans_df.columns:
        raise KeyError(f'Dealership {c} not found in transportation_costs.csv columns.')
for s in suppliers:
    if s not in trans_df.index:
        raise KeyError(f'Supplier {s} not found in transportation_costs.csv rows.')
transportation_costs = {}
for i in suppliers:
    for j in dealerships:
        transportation_costs[i, j] = float(trans_df.loc[i, j])
m = gp.Model('Colorado_Motor_Vehicle_UFLP')
x = m.addVars(suppliers, dealerships, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
y = m.addVars(suppliers, vtype=gp.GRB.BINARY, name='')
fixed_cost_term = gp.quicksum((fixed_costs[i] * y[i] for i in suppliers))
transport_cost_term = gp.quicksum((transportation_costs[i, j] * x[i, j] for i in suppliers for j in dealerships))
m.setObjective(fixed_cost_term + transport_cost_term, gp.GRB.MINIMIZE)
for j in dealerships:
    m.addConstr(gp.quicksum((x[i, j] for i in suppliers)) == demand[j], name=f'demand_{j}')
for i in suppliers:
    for j in dealerships:
        m.addConstr(x[i, j] <= demand[j] * y[i], name=f'link_{i}_{j}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f}')
    print('\n--- Supplier Opening Decisions ---')
    for i in suppliers:
        print(f"  Supplier {i}: {('OPEN' if y[i].X > 0.5 else 'CLOSED')} (y={int(round(y[i].X))})")
    print('\n--- Shipment Plan (vehicles from supplier to dealership) ---')
    for i in suppliers:
        for j in dealerships:
            if x[i, j].X > 0.001:
                print(f'  {x[i, j].X:.2f} vehicles from Supplier {i} to Dealership {j}')
else:
    print(f'No optimal solution found. Status: {m.status}')