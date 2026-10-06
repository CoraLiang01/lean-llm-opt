import gurobipy as gp
import pandas as pd
import numpy as np
cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP12/cost.csv'
demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP12/demand.csv'
cost_df = pd.read_csv(cost_path, sep=',')
demand_df = pd.read_csv(demand_path, sep=',')
plants = [str(p).strip() for p in cost_df['plant']]
customers = [str(c).strip() for c in demand_df['customer']]
fixed_cost = {str(row['plant']).strip(): float(row['fixed_cost']) for _, row in cost_df.iterrows()}
capacity = {str(row['plant']).strip(): float(row['capacity']) for _, row in cost_df.iterrows()}
transport_cost = {}
for _, row in cost_df.iterrows():
    plant = str(row['plant']).strip()
    for cust in customers:
        if cust not in cost_df.columns:
            raise KeyError(f"Customer '{cust}' not found as a column in cost.csv")
        transport_cost[plant, cust] = float(row[cust])
demand = {str(row['customer']).strip(): float(row['demand']) for _, row in demand_df.iterrows()}
if set(plants) != set(cost_df['plant'].astype(str).str.strip()):
    raise ValueError('Mismatch in plant identifiers between extracted set and cost.csv')
if set(customers) != set(demand_df['customer'].astype(str).str.strip()):
    raise ValueError('Mismatch in customer identifiers between extracted set and demand.csv')
if set(customers) - set(cost_df.columns):
    raise ValueError(f'Some customers in demand.csv are not columns in cost.csv: {set(customers) - set(cost_df.columns)}')
m = gp.Model('UFLP')
x = m.addVars(plants, customers, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
y = m.addVars(plants, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((fixed_cost[i] * y[i] for i in plants)) + gp.quicksum((transport_cost[i, j] * x[i, j] for i in plants for j in customers)), gp.GRB.MINIMIZE)
for j in customers:
    m.addConstr(gp.quicksum((x[i, j] for i in plants)) == demand[j], name=f'demand_{j}')
for i in plants:
    m.addConstr(gp.quicksum((x[i, j] for j in customers)) <= capacity[i] * y[i], name=f'capacity_{i}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f}')
    print('\n--- Plant Opening Decisions ---')
    for i in plants:
        status = 'OPEN' if y[i].X > 0.5 else 'CLOSED'
        print(f'Plant {i}: {status} (y={int(round(y[i].X))})')
    print('\n--- Shipment Plan (positive flows only) ---')
    for i in plants:
        for j in customers:
            shipped = x[i, j].X
            if shipped > 1e-06:
                print(f'  Plant {i} -> Customer {j}: {shipped:.2f}')
else:
    print(f'No optimal solution found. Status: {m.status}')