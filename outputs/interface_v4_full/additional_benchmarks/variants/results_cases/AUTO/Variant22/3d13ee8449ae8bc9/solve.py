import gurobipy as gp
import pandas as pd
import numpy as np
option_catalog_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant22/inputs/option_catalog.csv'
resource_limits_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant22/inputs/resource_limits.csv'
option_df = pd.read_csv(option_catalog_path, sep=',')
option_df['Family'] = option_df['Family'].astype(str)
option_df['Option'] = option_df['Option'].astype(str)
limits_df = pd.read_csv(resource_limits_path, sep=',')
limits_df['Resource'] = limits_df['Resource'].astype(str)
families = sorted(option_df['Family'].unique())
options_per_family = {fam: sorted(option_df.loc[option_df['Family'] == fam, 'Option'].unique()) for fam in families}
fam_opt_pairs = [(fam, opt) for fam in families for opt in options_per_family[fam]]
value = {(row['Family'], row['Option']): int(row['Value']) for _, row in option_df.iterrows()}
weight = {(row['Family'], row['Option']): int(row['Weight']) for _, row in option_df.iterrows()}
labor = {(row['Family'], row['Option']): int(row['LaborHours']) for _, row in option_df.iterrows()}
resource_limits = {row['Resource'].strip(): int(row['Limit']) for _, row in limits_df.iterrows()}
for res in ['Weight', 'LaborHours']:
    if res not in resource_limits:
        raise ValueError(f"Resource limit for '{res}' not found in resource_limits.csv.")
m = gp.Model('MultiChoiceKnapsack')
x = m.addVars(fam_opt_pairs, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((value[c, o] * x[c, o] for c, o in fam_opt_pairs)), gp.GRB.MAXIMIZE)
for fam in families:
    m.addConstr(gp.quicksum((x[fam, opt] for opt in options_per_family[fam])) == 1, name=f'OneOption_{fam}')
m.addConstr(gp.quicksum((weight[c, o] * x[c, o] for c, o in fam_opt_pairs)) <= resource_limits['Weight'], name='TotalWeight')
m.addConstr(gp.quicksum((labor[c, o] * x[c, o] for c, o in fam_opt_pairs)) <= resource_limits['LaborHours'], name='TotalLabor')
m.optimize()