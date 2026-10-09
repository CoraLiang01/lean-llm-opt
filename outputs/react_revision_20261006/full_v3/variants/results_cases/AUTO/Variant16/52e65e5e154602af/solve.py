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
options = sorted(option_df['Option'].unique())
famopt_tuples = [tuple(row) for row in option_df[['Family', 'Option']].values]
value = {(row['Family'], row['Option']): int(row['Value']) for (_, row) in option_df.iterrows()}
weight = {(row['Family'], row['Option']): int(row['Weight']) for (_, row) in option_df.iterrows()}
budgetuse = {(row['Family'], row['Option']): int(row['BudgetUse']) for (_, row) in option_df.iterrows()}
resource_limits = {}
for (_, row) in limits_df.iterrows():
    resource = row['Resource'].strip()
    resource_limits[resource] = int(row['Limit'])
for res in ['Weight', 'BudgetUse']:
    if res not in resource_limits:
        raise ValueError(f"Resource limit for '{res}' not found in resource_limits.csv.")
for key in famopt_tuples:
    if key not in value or key not in weight or key not in budgetuse:
        raise ValueError(f'Missing parameter for family-option pair {key}.')
family_to_options = {}
for fam in families:
    family_to_options[fam] = sorted(option_df[option_df['Family'] == fam]['Option'].unique())
m = gp.Model('multi_choice_knapsack')
m.setParam('MIPGap', 0.0001)
x = m.addVars(famopt_tuples, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((value[g, o] * x[g, o] for (g, o) in famopt_tuples)), gp.GRB.MAXIMIZE)
for g in families:
    m.addConstr(gp.quicksum((x[g, o] for o in family_to_options[g])) == 1, name='oneopt_' + g)
m.addConstr(gp.quicksum((weight[g, o] * x[g, o] for (g, o) in famopt_tuples)) <= resource_limits['Weight'], name='weight_limit')
m.addConstr(gp.quicksum((budgetuse[g, o] * x[g, o] for (g, o) in famopt_tuples)) <= resource_limits['BudgetUse'], name='budgetuse_limit')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for (g, o) in famopt_tuples:
        print(f'x[{g},{o}] {x[g, o].VarName} {x[g, o].X}')
else:
    print(f'Solver status: {m.status}')