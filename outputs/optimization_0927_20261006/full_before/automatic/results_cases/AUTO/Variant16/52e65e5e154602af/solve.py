import gurobipy as gp
import pandas as pd
import numpy as np
import re
option_catalog_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant16/inputs/option_catalog.csv'
resource_limits_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant16/inputs/resource_limits.csv'
option_df = pd.read_csv(option_catalog_path, sep=',')
resource_df = pd.read_csv(resource_limits_path, sep=',')
option_df['Family'] = option_df['Family'].astype(str).str.strip()
option_df['Option'] = option_df['Option'].astype(str).str.strip()
families = sorted(option_df['Family'].unique())
options_by_family = {g: sorted(option_df.loc[option_df['Family'] == g, 'Option'].unique()) for g in families}
family_option_pairs = [(g, o) for g in families for o in options_by_family[g]]
value = {}
weight = {}
budget_use = {}
for (_, row) in option_df.iterrows():
    g = str(row['Family']).strip()
    o = str(row['Option']).strip()
    value[g, o] = int(row['Value'])
    weight[g, o] = int(row['Weight'])
    budget_use[g, o] = int(row['BudgetUse'])
resource_df['Resource'] = resource_df['Resource'].astype(str).str.strip()
resource_limits = {}
for (_, row) in resource_df.iterrows():
    resource = row['Resource']
    limit = int(row['Limit'])
    resource_limits[resource] = limit
if 'Weight' not in resource_limits or 'BudgetUse' not in resource_limits:
    raise ValueError("Resource limits must include both 'Weight' and 'BudgetUse'.")
m = gp.Model('MultiChoiceKnapsack')
x = m.addVars(family_option_pairs, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((value[g, o] * x[g, o] for (g, o) in family_option_pairs)), gp.GRB.MAXIMIZE)
for g in families:
    m.addConstr(gp.quicksum((x[g, o] for o in options_by_family[g])) == 1, name=f'one_option_{g}')
m.addConstr(gp.quicksum((weight[g, o] * x[g, o] for (g, o) in family_option_pairs)) <= resource_limits['Weight'], name='total_weight')
m.addConstr(gp.quicksum((budget_use[g, o] * x[g, o] for (g, o) in family_option_pairs)) <= resource_limits['BudgetUse'], name='total_budget')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f}')
    print('--- Selected Options ---')
    for g in families:
        for o in options_by_family[g]:
            if x[g, o].X > 0.5:
                print(f'Family {g}: Option {o} (Value={value[g, o]}, Weight={weight[g, o]}, BudgetUse={budget_use[g, o]})')
    total_weight = sum((weight[g, o] * x[g, o].X for (g, o) in family_option_pairs))
    total_budget = sum((budget_use[g, o] * x[g, o].X for (g, o) in family_option_pairs))
    print(f"Total Weight Used: {total_weight:.0f} / {resource_limits['Weight']}")
    print(f"Total Budget Used: {total_budget:.0f} / {resource_limits['BudgetUse']}")
else:
    print(f'No optimal solution found. Status: {m.status}')