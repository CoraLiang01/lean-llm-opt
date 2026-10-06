import gurobipy as gp
import pandas as pd
import numpy as np
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant11/inputs/monthly_lot_sizing.csv'
df = pd.read_csv(csv_path, sep=',')
months = df['Month'].astype(str).tolist()
n_months = len(months)
demand = dict(zip(months, df['Demand'].astype(float)))
prod_cost = dict(zip(months, df['ProductionCost'].astype(float)))
setup_cost = dict(zip(months, df['SetupCost'].astype(float)))
hold_cost = dict(zip(months, df['HoldingCost'].astype(float)))
prod_cap = dict(zip(months, df['ProductionCapacity'].astype(float)))
for colname, param in [('Demand', demand), ('ProductionCost', prod_cost), ('SetupCost', setup_cost), ('HoldingCost', hold_cost), ('ProductionCapacity', prod_cap)]:
    if set(param.keys()) != set(months):
        raise ValueError(f'Missing {colname} data for some months.')

def solve_lot_sizing(months, demand, prod_cost, setup_cost, hold_cost, prod_cap):
    m = gp.Model('CapacitatedLotSizing')
    x = m.addVars(months, vtype=gp.GRB.CONTINUOUS, lb=0.0, name='')
    inv = m.addVars(months, vtype=gp.GRB.CONTINUOUS, lb=0.0, name='')
    y = m.addVars(months, vtype=gp.GRB.BINARY, name='')
    m.setObjective(gp.quicksum((prod_cost[t] * x[t] + setup_cost[t] * y[t] + hold_cost[t] * inv[t] for t in months)), gp.GRB.MINIMIZE)
    for idx, t in enumerate(months):
        if idx == 0:
            m.addConstr(x[t] - demand[t] == inv[t], name=f'inv_bal_{t}')
        else:
            prev = months[idx - 1]
            m.addConstr(inv[prev] + x[t] - demand[t] == inv[t], name=f'inv_bal_{t}')
    for t in months:
        m.addConstr(x[t] <= prod_cap[t] * y[t], name=f'cap_link_{t}')
    m.addConstr(inv[months[-1]] == 0, name='zero_end_inv')
    m.optimize()
    return m
m = solve_lot_sizing(months, demand, prod_cost, setup_cost, hold_cost, prod_cap)