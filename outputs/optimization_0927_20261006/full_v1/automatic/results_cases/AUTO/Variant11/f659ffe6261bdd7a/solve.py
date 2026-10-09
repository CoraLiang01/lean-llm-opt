import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant11/inputs/monthly_lot_sizing.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
months = df['Month'].tolist()
demand = {m: int(df.loc[df['Month'] == m, 'Demand'].values[0]) for m in months}
production_cost = {m: float(df.loc[df['Month'] == m, 'ProductionCost'].values[0]) for m in months}
setup_cost = {m: float(df.loc[df['Month'] == m, 'SetupCost'].values[0]) for m in months}
holding_cost = {m: float(df.loc[df['Month'] == m, 'HoldingCost'].values[0]) for m in months}
production_capacity = {m: float(df.loc[df['Month'] == m, 'ProductionCapacity'].values[0]) for m in months}
m = gp.Model('CapacitatedLotSizing')
x_vars = m.addVars(months, vtype=gp.GRB.CONTINUOUS, lb=0.0, name='')
inv_vars = m.addVars(months, vtype=gp.GRB.CONTINUOUS, lb=0.0, name='')
y_vars = m.addVars(months, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((production_cost[mth] * x_vars[mth] + setup_cost[mth] * y_vars[mth] + holding_cost[mth] * inv_vars[mth] for mth in months)), gp.GRB.MINIMIZE)
for (idx, mth) in enumerate(months):
    if idx == 0:
        m.addConstr(x_vars[mth] - demand[mth] == inv_vars[mth], name=f'inv_balance_{mth}')
    else:
        prev_mth = months[idx - 1]
        m.addConstr(inv_vars[prev_mth] + x_vars[mth] - demand[mth] == inv_vars[mth], name=f'inv_balance_{mth}')
last_month = months[-1]
m.addConstr(inv_vars[last_month] == 0, name='zero_ending_inventory')
for mth in months:
    m.addConstr(x_vars[mth] <= production_capacity[mth] * y_vars[mth], name=f'capacity_link_{mth}')
m.optimize()