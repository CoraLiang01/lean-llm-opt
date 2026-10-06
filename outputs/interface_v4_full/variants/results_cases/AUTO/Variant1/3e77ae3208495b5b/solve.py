import gurobipy as gp
import pandas as pd
import numpy as np

def solve_lot_sizing():
    path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant1/inputs/monthly_lot_sizing.csv'
    df = pd.read_csv(path, sep=',')
    required_cols = ['Month', 'Demand', 'ProductionCost', 'SetupCost', 'HoldingCost', 'ProductionCapacity']
    for col in required_cols:
        if col not in df.columns:
            raise KeyError(f"Required column '{col}' not found in CSV.")
    months = list(df['Month'])
    if len(months) != 24:
        raise ValueError(f'Expected 24 months, got {len(months)}.')
    demand = df.set_index('Month')['Demand'].to_dict()
    prod_cost = df.set_index('Month')['ProductionCost'].to_dict()
    setup_cost = df.set_index('Month')['SetupCost'].to_dict()
    hold_cost = df.set_index('Month')['HoldingCost'].to_dict()
    prod_cap = df.set_index('Month')['ProductionCapacity'].to_dict()
    m = gp.Model('LotSizing')
    x = m.addVars(months, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
    I = m.addVars(months, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
    y = m.addVars(months, vtype=gp.GRB.BINARY, name='')
    m.setObjective(gp.quicksum((prod_cost[mon] * x[mon] + setup_cost[mon] * y[mon] + hold_cost[mon] * I[mon] for mon in months)), gp.GRB.MINIMIZE)
    m.addConstr(x[months[0]] - demand[months[0]] == I[months[0]], name='invbal_1')
    for t in range(1, len(months)):
        prev = months[t - 1]
        curr = months[t]
        m.addConstr(I[prev] + x[curr] - demand[curr] == I[curr], name=f'invbal_{t + 1}')
    m.addConstr(I[months[-1]] == 0, name='final_inventory_zero')
    for mon in months:
        m.addConstr(x[mon] <= prod_cap[mon] * y[mon], name=f'cap_link_{mon}')
    m.optimize()
    return m
m = solve_lot_sizing()