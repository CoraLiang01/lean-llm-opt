import gurobipy as gp
import pandas as pd
import numpy as np
import re
option_catalog_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant16/inputs/option_catalog.csv'
resource_limits_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant16/inputs/resource_limits.csv'
option_df = pd.read_csv(option_catalog_path, sep=',', dtype=str, keep_default_na=False)
for col in ['Value', 'Weight', 'BudgetUse']:
    option_df[col] = option_df[col].astype(int)
limits_df = pd.read_csv(resource_limits_path, sep=',', dtype=str, keep_default_na=False)
limits_df['Limit'] = limits_df['Limit'].astype(int)
families = sorted(option_df['Family'].unique())
family_options = {g: sorted(option_df[option_df['Family'] == g]['Option'].unique()) for g in families}
go_pairs = [(g, o) for g in families for o in family_options[g]]
value = {(row['Family'], row['Option']): row['Value'] for (_, row) in option_df.iterrows()}
weight = {(row['Family'], row['Option']): row['Weight'] for (_, row) in option_df.iterrows()}
budget_use = {(row['Family'], row['Option']): row['BudgetUse'] for (_, row) in option_df.iterrows()}
limits_dict = {row['Resource'].strip(): row['Limit'] for (_, row) in limits_df.iterrows()}
for res in ['Weight', 'BudgetUse']:
    if res not in limits_dict:
        raise ValueError(f"Resource limit for '{res}' not found in resource_limits.csv.")
weight_limit = limits_dict['Weight']
budget_limit = limits_dict['BudgetUse']
m = gp.Model('MultiChoiceKnapsack')
x_vars = m.addVars(go_pairs, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((value[g, o] * x_vars[g, o] for (g, o) in go_pairs)), gp.GRB.MAXIMIZE)
for g in families:
    m.addConstr(gp.quicksum((x_vars[g, o] for o in family_options[g])) == 1, name=f'OneOption_{g}')
m.addConstr(gp.quicksum((weight[g, o] * x_vars[g, o] for (g, o) in go_pairs)) <= weight_limit, name='TotalWeight')
m.addConstr(gp.quicksum((budget_use[g, o] * x_vars[g, o] for (g, o) in go_pairs)) <= budget_limit, name='TotalBudgetUse')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value: {m.objVal:.0f}')
    print('--- Selected Options ---')
    for g in families:
        for o in family_options[g]:
            if x_vars[g, o].X > 0.5:
                print(f'Family {g}: Option {o} (Value={value[g, o]}, Weight={weight[g, o]}, BudgetUse={budget_use[g, o]})')
    total_weight = sum((weight[g, o] * x_vars[g, o].X for (g, o) in go_pairs))
    total_budget = sum((budget_use[g, o] * x_vars[g, o].X for (g, o) in go_pairs))
    print(f'Total Weight Used: {total_weight:.0f} / {weight_limit}')
    print(f'Total Budget Used: {total_budget:.0f} / {budget_limit}')
else:
    print(f'No optimal solution found. Status: {m.status}')