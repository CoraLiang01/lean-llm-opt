import gurobipy as gp
import pandas as pd
import numpy as np
cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP12/cost.csv'
demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP12/demand.csv'
cost_df = pd.read_csv(cost_path, sep=',')
demand_df = pd.read_csv(demand_path, sep=',')
plants = cost_df['plant'].astype(str).tolist()
customers = demand_df['customer'].astype(str).tolist()
cost_customer_cols = [c for c in cost_df.columns if c.startswith('C')]
if set(customers) != set(cost_customer_cols):
    raise ValueError(f'Mismatch between customers in demand.csv and columns in cost.csv: {set(customers) ^ set(cost_customer_cols)}')
fixed_cost = dict(zip(cost_df['plant'].astype(str), cost_df['fixed_cost']))
capacity = dict(zip(cost_df['plant'].astype(str), cost_df['capacity']))
transport_cost = {}
for i, row in cost_df.iterrows():
    plant = str(row['plant'])
    transport_cost[plant] = {}
    for cust in customers:
        transport_cost[plant][cust] = float(row[cust])
demand = dict(zip(demand_df['customer'].astype(str), demand_df['demand']))
m = gp.Model('CapacitatedFacilityLocation')
x = m.addVars(plants, customers, lb=0.0, name='')
y = m.addVars(plants, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((fixed_cost[i] * y[i] for i in plants)) + gp.quicksum((transport_cost[i][j] * x[i, j] for i in plants for j in customers)), gp.GRB.MINIMIZE)
m.addConstrs((gp.quicksum((x[i, j] for i in plants)) == demand[j] for j in customers), name='')
m.addConstrs((gp.quicksum((x[i, j] for j in customers)) <= capacity[i] * y[i] for i in plants), name='')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f}')
    print('--- Plant Opening Decisions ---')
    for i in plants:
        print(f"Plant {i}: {('OPEN' if y[i].X > 0.5 else 'CLOSED')} (y={int(round(y[i].X))})")
    print('--- Shipment Plan (positive flows only) ---')
    for i in plants:
        for j in customers:
            if x[i, j].X > 1e-06:
                print(f'  Ship {x[i, j].X:.2f} units from {i} to {j} (cost per unit: {transport_cost[i][j]})')
else:
    print(f'No optimal solution found. Status: {m.status}')