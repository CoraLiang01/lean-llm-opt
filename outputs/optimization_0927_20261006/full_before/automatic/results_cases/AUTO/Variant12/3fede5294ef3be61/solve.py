import gurobipy as gp
import pandas as pd
import numpy as np
plant_cap_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant12/inputs/plant_capacity.csv', sep=',')
retailer_dem_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant12/inputs/retailer_demand.csv', sep=',')
route_varcost_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant12/inputs/route_variable_costs.csv', sep=',')
route_fixedcost_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant12/inputs/route_fixed_costs.csv', sep=',')
plants = plant_cap_df['Plant'].astype(str).tolist()
retailers = retailer_dem_df['Retailer'].astype(str).tolist()
plant_capacity = plant_cap_df.set_index('Plant')['SupplyCapacity'].astype(int).to_dict()
retailer_demand = retailer_dem_df.set_index('Retailer')['Demand'].astype(int).to_dict()
route_varcost_df['Plant'] = route_varcost_df['Plant'].astype(str)
c = {}
for (_, row) in route_varcost_df.iterrows():
    i = row['Plant']
    for j in retailers:
        c[i, j] = int(row[j])
route_fixedcost_df['Plant'] = route_fixedcost_df['Plant'].astype(str)
f = {}
for (_, row) in route_fixedcost_df.iterrows():
    i = row['Plant']
    for j in retailers:
        f[i, j] = int(row[j])
M = {}
for i in plants:
    for j in retailers:
        M[i, j] = min(plant_capacity[i], retailer_demand[j])
for i in plants:
    for j in retailers:
        if (i, j) not in c:
            raise KeyError(f'Missing variable cost for route ({i},{j})')
        if (i, j) not in f:
            raise KeyError(f'Missing fixed cost for route ({i},{j})')
        if (i, j) not in M:
            raise KeyError(f'Missing M value for route ({i},{j})')
m = gp.Model('FixedChargeTransportation')
x = m.addVars(plants, retailers, vtype=gp.GRB.CONTINUOUS, lb=0.0, name='')
y = m.addVars(plants, retailers, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((c[i, j] * x[i, j] + f[i, j] * y[i, j] for i in plants for j in retailers)), gp.GRB.MINIMIZE)
for j in retailers:
    m.addConstr(gp.quicksum((x[i, j] for i in plants)) == retailer_demand[j], name=f'demand_{j}')
for i in plants:
    m.addConstr(gp.quicksum((x[i, j] for j in retailers)) <= plant_capacity[i], name=f'supply_{i}')
for i in plants:
    for j in retailers:
        m.addConstr(x[i, j] <= M[i, j] * y[i, j], name=f'link_{i}_{j}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total cost: {m.objVal:.2f}')
    print('\n--- Shipment Plan (x[i,j]) ---')
    for i in plants:
        for j in retailers:
            if x[i, j].X > 1e-06:
                print(f'  Ship {x[i, j].X:.2f} units from {i} to {j} (Route used: {int(y[i, j].X)})')
    print('\n--- Route Activation (y[i,j]) ---')
    for i in plants:
        for j in retailers:
            print(f"  Route {i}->{j}: {('OPEN' if y[i, j].X > 0.5 else 'closed')}")
else:
    print(f'No optimal solution found. Status: {m.status}')