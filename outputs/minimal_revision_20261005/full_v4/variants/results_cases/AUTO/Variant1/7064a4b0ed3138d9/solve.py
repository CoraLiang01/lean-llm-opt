import gurobipy as gp
import pandas as pd
import numpy as np
import re

def solve_lot_sizing():
    csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant1/inputs/monthly_lot_sizing.csv'
    df = pd.read_csv(csv_path, sep=',')
    required_cols = ['Month', 'Demand', 'ProductionCost', 'SetupCost', 'HoldingCost', 'ProductionCapacity']
    for col in required_cols:
        if col not in df.columns:
            raise KeyError(f'Missing required column: {col}')
    months = list(df['Month'])
    n_months = len(months)
    if n_months != 24:
        raise ValueError(f'Expected 24 months, got {n_months}')
    demand = dict(zip(months, df['Demand'].astype(float)))
    prod_cost = dict(zip(months, df['ProductionCost'].astype(float)))
    setup_cost = dict(zip(months, df['SetupCost'].astype(float)))
    hold_cost = dict(zip(months, df['HoldingCost'].astype(float)))
    prod_cap = dict(zip(months, df['ProductionCapacity'].astype(float)))
    for m in months:
        for (dct, name) in [(demand, 'Demand'), (prod_cost, 'ProductionCost'), (setup_cost, 'SetupCost'), (hold_cost, 'HoldingCost'), (prod_cap, 'ProductionCapacity')]:
            if m not in dct:
                raise KeyError(f'Missing {name} for month {m}')
    I_idx = list(range(n_months + 1))
    I_month_map = {i + 1: months[i] for i in range(n_months)}
    m = gp.Model('CapacitatedLotSizing')
    x = m.addVars(months, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
    y = m.addVars(months, vtype=gp.GRB.BINARY, name='')
    I = m.addVars(I_idx, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
    m.addConstr(I[0] == 0, name='init_inventory')
    for (t_idx, mth) in enumerate(months):
        m.addConstr(I[t_idx + 1] == I[t_idx] + x[mth] - demand[mth], name=f'inv_bal_{mth}')
    m.addConstr(I[n_months] == 0, name='final_inventory')
    for mth in months:
        m.addConstr(x[mth] <= prod_cap[mth] * y[mth], name=f'prod_cap_{mth}')
    total_cost = gp.quicksum((prod_cost[mth] * x[mth] for mth in months)) + gp.quicksum((setup_cost[mth] * y[mth] for mth in months)) + gp.quicksum((hold_cost[mth] * I[t_idx + 1] for (t_idx, mth) in enumerate(months)))
    m.setObjective(total_cost, gp.GRB.MINIMIZE)
    m.setParam('MIPGap', 0.0001)
    m.optimize()
    if m.status == gp.GRB.OPTIMAL:
        print(f'ObjVal {m.objVal:.6f}')
        for mth in months:
            print(f'x[{mth}] {x[mth].VarName} {x[mth].X:.6f}')
        for mth in months:
            print(f'y[{mth}] {y[mth].VarName} {y[mth].X:.0f}')
        for i in I_idx:
            print(f'I[{i}] {I[i].VarName} {I[i].X:.6f}')
    else:
        print(f'Solver status: {m.status}')
    return m
m = solve_lot_sizing()