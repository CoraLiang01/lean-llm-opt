import gurobipy as gp
import pandas as pd
import numpy as np
option_catalog_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant22/inputs/option_catalog.csv'
resource_limits_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant22/inputs/resource_limits.csv'
option_df = pd.read_csv(option_catalog_path, sep=',')
option_df['Family'] = option_df['Family'].astype(str).str.strip()
option_df['Option'] = option_df['Option'].astype(str).str.strip()
limits_df = pd.read_csv(resource_limits_path, sep=',')
limits_df['Resource'] = limits_df['Resource'].astype(str).str.strip()
families = sorted(option_df['Family'].unique())
options_per_family = {fam: sorted(option_df.loc[option_df['Family'] == fam, 'Option'].unique()) for fam in families}
all_options = sorted(option_df['Option'].unique())
value = {}
weight = {}
laborhours = {}
for _, row in option_df.iterrows():
    c = row['Family']
    o = row['Option']
    value[c, o] = int(row['Value'])
    weight[c, o] = int(row['Weight'])
    laborhours[c, o] = int(row['LaborHours'])
resource_limits = {}
for _, row in limits_df.iterrows():
    resource = row['Resource']
    resource_limits[resource] = int(row['Limit'])
for res in ['Weight', 'LaborHours']:
    if res not in resource_limits:
        raise ValueError(f"Resource limit for '{res}' not found in resource_limits.csv.")
m = gp.Model('MultiChoiceKnapsack')
x = {}
for c in families:
    for o in options_per_family[c]:
        x[c, o] = m.addVar(vtype=gp.GRB.BINARY, name=f'x_{c}_{o}')
m.setObjective(gp.quicksum((value[c, o] * x[c, o] for c in families for o in options_per_family[c])), gp.GRB.MAXIMIZE)
for c in families:
    m.addConstr(gp.quicksum((x[c, o] for o in options_per_family[c])) == 1, name=f'OneOption_{c}')
m.addConstr(gp.quicksum((weight[c, o] * x[c, o] for c in families for o in options_per_family[c])) <= resource_limits['Weight'], name='WeightLimit')
m.addConstr(gp.quicksum((laborhours[c, o] * x[c, o] for c in families for o in options_per_family[c])) <= resource_limits['LaborHours'], name='LaborHoursLimit')
m.optimize()