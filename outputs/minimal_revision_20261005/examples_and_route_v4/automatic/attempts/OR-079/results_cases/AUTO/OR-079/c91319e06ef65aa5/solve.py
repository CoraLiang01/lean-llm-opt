import gurobipy as gp
import pandas as pd
import numpy as np
facility_costs_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP11/facility_costs.csv'
shipping_costs_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP11/shipping_costs.csv'
demand_requirements_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP11/demand_requirements.csv'
fac_df = pd.read_csv(facility_costs_path, sep=',')
fac_df['Facility'] = fac_df['Facility'].astype(str).str.strip()
facilities = fac_df['Facility'].tolist()
dem_df = pd.read_csv(demand_requirements_path, sep=',')
dem_df['Destination'] = dem_df['Destination'].astype(str).str.strip()
dcs = dem_df['Destination'].tolist()
ship_df = pd.read_csv(shipping_costs_path, sep=',')
ship_df['Origin'] = ship_df['Origin'].astype(str).str.strip()
if set(facilities) != set(ship_df['Origin']):
    raise ValueError('Mismatch between facilities in facility_costs.csv and shipping_costs.csv')
if set(dcs) != set([col for col in ship_df.columns if col != 'Origin']):
    raise ValueError('Mismatch between DCs in demand_requirements.csv and shipping_costs.csv')
fixed_cost = dict(zip(fac_df['Facility'], fac_df['FixedCost']))
capacity = dict(zip(fac_df['Facility'], fac_df['Capacity']))
demand = dict(zip(dem_df['Destination'], dem_df['Demand']))
shipping_cost = {}
for (_, row) in ship_df.iterrows():
    i = str(row['Origin']).strip()
    for j in dcs:
        shipping_cost[i, j] = float(row[j])
for i in facilities:
    if i not in fixed_cost or i not in capacity:
        raise ValueError(f'Missing facility {i} in fixed_cost or capacity')
for j in dcs:
    if j not in demand:
        raise ValueError(f'Missing DC {j} in demand')
for i in facilities:
    for j in dcs:
        if (i, j) not in shipping_cost:
            raise ValueError(f'Missing shipping cost for ({i},{j})')

def solve_uflp(facilities, dcs, fixed_cost, capacity, demand, shipping_cost):
    m = gp.Model('UFLP')
    m.Params.MIPGap = 0.0001
    y = m.addVars(facilities, vtype=gp.GRB.BINARY, name='')
    x = m.addVars([(i, j) for i in facilities for j in dcs], lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
    m.setObjective(gp.quicksum((fixed_cost[i] * y[i] for i in facilities)) + gp.quicksum((shipping_cost[i, j] * x[i, j] for i in facilities for j in dcs)), gp.GRB.MINIMIZE)
    m.addConstrs((gp.quicksum((x[i, j] for i in facilities)) == demand[j] for j in dcs), name='')
    m.addConstrs((gp.quicksum((x[i, j] for j in dcs)) <= capacity[i] * y[i] for i in facilities), name='')
    m.optimize()
    return m
m = solve_uflp(facilities, dcs, fixed_cost, capacity, demand, shipping_cost)
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')