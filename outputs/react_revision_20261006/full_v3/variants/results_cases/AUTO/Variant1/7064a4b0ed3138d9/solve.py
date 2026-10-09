import gurobipy as gp
import pandas as pd
import numpy as np
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant1/inputs/monthly_lot_sizing.csv'
df = pd.read_csv(csv_path, sep=',')
months = list(df['Month'])
if len(months) != 24 or len(set(months)) != 24:
    raise ValueError('Expected 24 unique months in the CSV file.')
demand = df.set_index('Month')['Demand'].to_dict()
prod_cost = df.set_index('Month')['ProductionCost'].to_dict()
setup_cost = df.set_index('Month')['SetupCost'].to_dict()
hold_cost = df.set_index('Month')['HoldingCost'].to_dict()
prod_cap = df.set_index('Month')['ProductionCapacity'].to_dict()
for m in months:
    for (dct, name) in [(demand, 'Demand'), (prod_cost, 'ProductionCost'), (setup_cost, 'SetupCost'), (hold_cost, 'HoldingCost'), (prod_cap, 'ProductionCapacity')]:
        if m not in dct:
            raise ValueError(f'Missing {name} for month {m}.')

def solve_lot_sizing(months, demand, prod_cost, setup_cost, hold_cost, prod_cap):
    m = gp.Model('LotSizing')
    m.setParam('MIPGap', 0.0001)
    x = m.addVars(months, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
    I = m.addVars(months, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
    y = m.addVars(months, vtype=gp.GRB.BINARY, name='')
    m.setObjective(gp.quicksum((prod_cost[mth] * x[mth] + setup_cost[mth] * y[mth] + hold_cost[mth] * I[mth] for mth in months)), gp.GRB.MINIMIZE)
    m.addConstr(x[months[0]] - demand[months[0]] == I[months[0]], name='invbal_1')
    for t in range(1, len(months)):
        prev = months[t - 1]
        curr = months[t]
        m.addConstr(I[prev] + x[curr] - demand[curr] == I[curr], name=f'invbal_{curr}')
    for mth in months:
        m.addConstr(x[mth] <= prod_cap[mth] * y[mth], name=f'caplink_{mth}')
    m.addConstr(I[months[-1]] == 0, name='final_inventory_zero')
    m.optimize()
    return m
m = solve_lot_sizing(months, demand, prod_cost, setup_cost, hold_cost, prod_cap)
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal {m.objVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')