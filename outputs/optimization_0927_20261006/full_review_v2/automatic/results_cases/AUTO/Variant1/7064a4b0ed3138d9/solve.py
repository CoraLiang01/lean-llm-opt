import gurobipy as gp
import pandas as pd
import numpy as np
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant1/inputs/monthly_lot_sizing.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
months = list(df['Month'])
demand = pd.Series(df['Demand'].astype(float).values, index=months)
production_cost = pd.Series(df['ProductionCost'].astype(float).values, index=months)
setup_cost = pd.Series(df['SetupCost'].astype(float).values, index=months)
holding_cost = pd.Series(df['HoldingCost'].astype(float).values, index=months)
production_capacity = pd.Series(df['ProductionCapacity'].astype(float).values, index=months)
for (colname, series) in [('Demand', demand), ('ProductionCost', production_cost), ('SetupCost', setup_cost), ('HoldingCost', holding_cost), ('ProductionCapacity', production_capacity)]:
    if len(series) != len(months):
        raise ValueError(f'Column {colname} does not cover all months.')
    if series.isnull().any():
        raise ValueError(f'Column {colname} contains missing values.')
m = gp.Model('MonthlyLotSizing')
x_vars = m.addVars(months, vtype=gp.GRB.CONTINUOUS, lb=0.0, name='')
I_vars = m.addVars(months, vtype=gp.GRB.CONTINUOUS, lb=0.0, name='')
y_vars = m.addVars(months, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((production_cost[month] * x_vars[month] + setup_cost[month] * y_vars[month] + holding_cost[month] * I_vars[month] for month in months)), gp.GRB.MINIMIZE)
for (idx, month) in enumerate(months):
    if idx == 0:
        m.addConstr(I_vars[month] == x_vars[month] - demand[month], name=f'inv_bal_{month}')
    else:
        prev_month = months[idx - 1]
        m.addConstr(I_vars[month] == I_vars[prev_month] + x_vars[month] - demand[month], name=f'inv_bal_{month}')
for month in months:
    m.addConstr(x_vars[month] <= production_capacity[month] * y_vars[month], name=f'prod_cap_{month}')
m.addConstr(I_vars[months[-1]] == 0, name='final_zero_inventory')
m.optimize()