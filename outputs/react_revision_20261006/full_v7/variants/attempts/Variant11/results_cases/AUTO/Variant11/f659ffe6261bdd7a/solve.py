import gurobipy as gp
import pandas as pd
import numpy as np
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant11/inputs/monthly_lot_sizing.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
months = df['Month'].astype(str).str.strip().tolist()
n_months = len(months)

def to_numeric_series(colname, dtype):
    s = pd.to_numeric(df[colname], errors='raise')
    if len(s) != n_months:
        raise ValueError(f'Column {colname} length mismatch with months')
    return dict(zip(months, s.astype(dtype)))
demand = to_numeric_series('Demand', int)
prod_cost = to_numeric_series('ProductionCost', float)
setup_cost = to_numeric_series('SetupCost', float)
hold_cost = to_numeric_series('HoldingCost', float)
prod_cap = to_numeric_series('ProductionCapacity', float)
for m in months:
    for (param, d) in [('Demand', demand), ('ProductionCost', prod_cost), ('SetupCost', setup_cost), ('HoldingCost', hold_cost), ('ProductionCapacity', prod_cap)]:
        if m not in d:
            raise KeyError(f'Missing {param} for month {m}')

def solve_lot_sizing(months, demand, prod_cost, setup_cost, hold_cost, prod_cap):
    m = gp.Model('CapacitatedLotSizing')
    x_vars = m.addVars(months, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
    inv_vars = m.addVars(months, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
    y_vars = m.addVars(months, vtype=gp.GRB.BINARY, name='')
    m.setObjective(gp.quicksum((prod_cost[mth] * x_vars[mth] + setup_cost[mth] * y_vars[mth] + hold_cost[mth] * inv_vars[mth] for mth in months)), gp.GRB.MINIMIZE)
    for (idx, mth) in enumerate(months):
        if idx == 0:
            m.addConstr(x_vars[mth] - demand[mth] == inv_vars[mth], name=f'inv_bal_{mth}')
        else:
            prev_mth = months[idx - 1]
            m.addConstr(inv_vars[prev_mth] + x_vars[mth] - demand[mth] == inv_vars[mth], name=f'inv_bal_{mth}')
    m.addConstr(inv_vars[months[-1]] == 0, name='final_inventory_zero')
    for mth in months:
        m.addConstr(x_vars[mth] <= prod_cap[mth] * y_vars[mth], name=f'cap_link_{mth}')
    m.Params.MIPGap = 0.0001
    m.optimize()
    return m
m = solve_lot_sizing(months, demand, prod_cost, setup_cost, hold_cost, prod_cap)
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal:.2f}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X:.6f}')
else:
    print(f'Solver status: {m.status}')