import gurobipy as gp
import pandas as pd
import numpy as np
import re
option_catalog_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant6/inputs/option_catalog.csv'
resource_limits_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant6/inputs/resource_limits.csv'
option_df = pd.read_csv(option_catalog_path, sep=',', dtype=str, keep_default_na=False)
resource_df = pd.read_csv(resource_limits_path, sep=',', dtype=str, keep_default_na=False)
for col in ['Value', 'Weight', 'BudgetUse']:
    option_df[col] = option_df[col].astype(int)
resource_df['Limit'] = resource_df['Limit'].astype(int)
families = sorted(option_df['Family'].unique())
family_to_options = {fam: list(option_df.loc[option_df['Family'] == fam, 'Option']) for fam in families}
value = {}
weight = {}
budgetuse = {}
for (idx, row) in option_df.iterrows():
    f = row['Family']
    o = row['Option']
    value[f, o] = row['Value']
    weight[f, o] = row['Weight']
    budgetuse[f, o] = row['BudgetUse']
resource_limits = {}
for (idx, row) in resource_df.iterrows():
    resource = row['Resource'].strip()
    resource_limits[resource] = row['Limit']
for req in ['Weight', 'BudgetUse']:
    if req not in resource_limits:
        raise ValueError(f"Resource limit for '{req}' not found in resource_limits.csv")
m = gp.Model('MultiChoiceKnapsack')
x_vars = m.addVars([(f, o) for f in families for o in family_to_options[f]], vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((value[f, o] * x_vars[f, o] for f in families for o in family_to_options[f])), gp.GRB.MAXIMIZE)
for f in families:
    m.addConstr(gp.quicksum((x_vars[f, o] for o in family_to_options[f])) == 1, name=f'select_one_{f}')
m.addConstr(gp.quicksum((weight[f, o] * x_vars[f, o] for f in families for o in family_to_options[f])) <= resource_limits['Weight'], name='weight_limit')
m.addConstr(gp.quicksum((budgetuse[f, o] * x_vars[f, o] for f in families for o in family_to_options[f])) <= resource_limits['BudgetUse'], name='budget_limit')
m.optimize()