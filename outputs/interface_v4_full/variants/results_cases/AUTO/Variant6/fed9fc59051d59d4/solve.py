import gurobipy as gp
import pandas as pd
import numpy as np
option_catalog_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant6/inputs/option_catalog.csv'
option_df = pd.read_csv(option_catalog_path, sep=',')
resource_limits_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant6/inputs/resource_limits.csv'
limits_df = pd.read_csv(resource_limits_path, sep=',')
option_df['Family'] = option_df['Family'].astype(str).str.strip()
option_df['Option'] = option_df['Option'].astype(str).str.strip()
families = sorted(option_df['Family'].unique())
options_per_family = {f: sorted(option_df.loc[option_df['Family'] == f, 'Option'].unique()) for f in families}
fam_opt_pairs = [(row['Family'], row['Option']) for _, row in option_df.iterrows()]
value = {(row['Family'], row['Option']): int(row['Value']) for _, row in option_df.iterrows()}
weight = {(row['Family'], row['Option']): int(row['Weight']) for _, row in option_df.iterrows()}
budget_use = {(row['Family'], row['Option']): int(row['BudgetUse']) for _, row in option_df.iterrows()}
limits_df['Resource'] = limits_df['Resource'].astype(str).str.strip()
resource_limits = {row['Resource']: int(row['Limit']) for _, row in limits_df.iterrows()}
required_resources = ['Weight', 'BudgetUse']
for res in required_resources:
    if res not in resource_limits:
        raise ValueError(f"Resource limit for '{res}' not found in resource_limits.csv.")
m = gp.Model('MultiChoiceKnapsack')
x = m.addVars(fam_opt_pairs, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((value[fo] * x[fo] for fo in fam_opt_pairs)), gp.GRB.MAXIMIZE)
for f in families:
    m.addConstr(gp.quicksum((x[f, o] for o in options_per_family[f])) == 1, name=f'select_one_{f}')
m.addConstr(gp.quicksum((weight[fo] * x[fo] for fo in fam_opt_pairs)) <= resource_limits['Weight'], name='weight_limit')
m.addConstr(gp.quicksum((budget_use[fo] * x[fo] for fo in fam_opt_pairs)) <= resource_limits['BudgetUse'], name='budget_limit')
m.optimize()