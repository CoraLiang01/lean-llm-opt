import gurobipy as gp
import pandas as pd
import numpy as np
cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP12/cost.csv'
demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP12/demand.csv'
cost_df = pd.read_csv(cost_path, sep=',')
demand_df = pd.read_csv(demand_path, sep=',')
plants = cost_df['plant'].astype(str).tolist()
customers = demand_df['customer'].astype(str).tolist()
fixed_cost = cost_df.set_index('plant')['fixed_cost'].astype(float).to_dict()
capacity = cost_df.set_index('plant')['capacity'].astype(float).to_dict()
demand = demand_df.set_index('customer')['demand'].astype(float).to_dict()
transport_cost = {}
for (i, row) in cost_df.iterrows():
    plant = str(row['plant'])
    transport_cost[plant] = {}
    for cust in customers:
        if cust not in cost_df.columns:
            raise KeyError(f"Customer '{cust}' not found as a column in cost.csv")
        transport_cost[plant][cust] = float(row[cust])
if set(plants) != set(cost_df['plant'].astype(str)):
    raise ValueError('Mismatch in plant identifiers between cost.csv and extracted plant set.')
if set(customers) != set(demand_df['customer'].astype(str)):
    raise ValueError('Mismatch in customer identifiers between demand.csv and extracted customer set.')
for plant in plants:
    if plant not in fixed_cost or plant not in capacity or plant not in transport_cost:
        raise KeyError(f"Missing data for plant '{plant}'")
    for cust in customers:
        if cust not in transport_cost[plant]:
            raise KeyError(f"Missing transport cost for plant '{plant}', customer '{cust}'")
for cust in customers:
    if cust not in demand:
        raise KeyError(f"Missing demand for customer '{cust}'")
m = gp.Model('CapacitatedFacilityLocation')
x = m.addVars(plants, customers, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
y = m.addVars(plants, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((fixed_cost[i] * y[i] for i in plants)) + gp.quicksum((transport_cost[i][j] * x[i, j] for i in plants for j in customers)), gp.GRB.MINIMIZE)
m.addConstrs((gp.quicksum((x[i, j] for i in plants)) == demand[j] for j in customers), name='')
m.addConstrs((gp.quicksum((x[i, j] for j in customers)) <= capacity[i] * y[i] for i in plants), name='')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total cost: {m.objVal:.2f}')
    print('\n--- Plant Opening Decisions ---')
    for i in plants:
        status = 'OPEN' if y[i].X > 0.5 else 'CLOSED'
        print(f'Plant {i}: {status} (y={int(round(y[i].X))})')
    print('\n--- Shipment Plan (amounts > 0) ---')
    for i in plants:
        for j in customers:
            shipped = x[i, j].X
            if shipped > 1e-06:
                print(f'  Plant {i} -> Customer {j}: {shipped:.2f}')
else:
    print(f'No optimal solution found. Status: {m.status}')