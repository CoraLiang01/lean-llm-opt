import gurobipy as gp
import pandas as pd
import numpy as np
option_catalog_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant16/inputs/option_catalog.csv'
resource_limits_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant16/inputs/resource_limits.csv'
option_df = pd.read_csv(option_catalog_path, sep=',')
option_df['Family'] = option_df['Family'].astype(str).str.strip()
option_df['Option'] = option_df['Option'].astype(str).str.strip()
limits_df = pd.read_csv(resource_limits_path, sep=',')
limits_df['Resource'] = limits_df['Resource'].astype(str).str.strip()
families = sorted(option_df['Family'].unique())
family_options = {g: sorted(option_df.loc[option_df['Family'] == g, 'Option'].unique()) for g in families}
go_pairs = [(row['Family'], row['Option']) for (_, row) in option_df.iterrows()]
value = {(row['Family'], row['Option']): int(row['Value']) for (_, row) in option_df.iterrows()}
weight = {(row['Family'], row['Option']): int(row['Weight']) for (_, row) in option_df.iterrows()}
budget_use = {(row['Family'], row['Option']): int(row['BudgetUse']) for (_, row) in option_df.iterrows()}
resource_limits = {}
for (_, row) in limits_df.iterrows():
    resource = row['Resource']
    limit = int(row['Limit'])
    resource_limits[resource] = limit
for res in ['Weight', 'BudgetUse']:
    if res not in resource_limits:
        raise ValueError(f"Resource limit for '{res}' not found in resource_limits.csv.")
m = gp.Model('MultiChoiceKnapsack')
x = m.addVars(go_pairs, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((value[g, o] * x[g, o] for (g, o) in go_pairs)), gp.GRB.MAXIMIZE)
for g in families:
    m.addConstr(gp.quicksum((x[g, o] for o in family_options[g])) == 1, name=f'one_option_{g}')
m.addConstr(gp.quicksum((weight[g, o] * x[g, o] for (g, o) in go_pairs)) <= resource_limits['Weight'], name='weight_limit')
m.addConstr(gp.quicksum((budget_use[g, o] * x[g, o] for (g, o) in go_pairs)) <= resource_limits['BudgetUse'], name='budgetuse_limit')
m.setParam('MIPGap', 0.0001)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for (g, o) in go_pairs:
        print(f'{x[g, o].VarName} {x[g, o].X}')
else:
    print(f'Solver status: {m.status}')