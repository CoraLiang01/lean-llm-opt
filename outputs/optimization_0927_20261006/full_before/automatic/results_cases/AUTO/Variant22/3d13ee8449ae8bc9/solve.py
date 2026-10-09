import gurobipy as gp
import pandas as pd
import numpy as np
import re
option_catalog_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant22/inputs/option_catalog.csv'
resource_limits_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant22/inputs/resource_limits.csv'
option_df = pd.read_csv(option_catalog_path, sep=',')
resource_df = pd.read_csv(resource_limits_path, sep=',')
families = sorted(option_df['Family'].astype(str).unique())
options_per_family = {fam: sorted(option_df.loc[option_df['Family'].astype(str) == fam, 'Option'].astype(str).unique()) for fam in families}
fam_opt_pairs = [(row['Family'], row['Option']) for (_, row) in option_df.iterrows()]
value = {}
weight = {}
labor = {}
for (_, row) in option_df.iterrows():
    c = str(row['Family'])
    o = str(row['Option'])
    value[c, o] = int(row['Value'])
    weight[c, o] = int(row['Weight'])
    labor[c, o] = int(row['LaborHours'])
resource_limits = {}
for (_, row) in resource_df.iterrows():
    res = str(row['Resource']).strip()
    lim = int(row['Limit'])
    resource_limits[res] = lim
required_resources = ['Weight', 'LaborHours']
for res in required_resources:
    if res not in resource_limits:
        raise ValueError(f"Resource limit for '{res}' not found in resource_limits.csv.")
m = gp.Model('MultiChoiceKnapsack')
x = m.addVars(fam_opt_pairs, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((value[c, o] * x[c, o] for (c, o) in fam_opt_pairs)), gp.GRB.MAXIMIZE)
for c in families:
    m.addConstr(gp.quicksum((x[c, o] for o in options_per_family[c])) == 1, name=f'one_option_{c}')
m.addConstr(gp.quicksum((weight[c, o] * x[c, o] for (c, o) in fam_opt_pairs)) <= resource_limits['Weight'], name='weight_limit')
m.addConstr(gp.quicksum((labor[c, o] * x[c, o] for (c, o) in fam_opt_pairs)) <= resource_limits['LaborHours'], name='labor_limit')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f}')
    print('--- Selected Options ---')
    for c in families:
        for o in options_per_family[c]:
            if x[c, o].X > 0.5:
                print(f'Family {c}: Option {o} (Value={value[c, o]}, Weight={weight[c, o]}, LaborHours={labor[c, o]})')
    total_weight = sum((weight[c, o] * x[c, o].X for (c, o) in fam_opt_pairs))
    total_labor = sum((labor[c, o] * x[c, o].X for (c, o) in fam_opt_pairs))
    print(f"Total Weight Used: {total_weight:.0f} / {resource_limits['Weight']}")
    print(f"Total LaborHours Used: {total_labor:.0f} / {resource_limits['LaborHours']}")
else:
    print(f'No optimal solution found. Status: {m.status}')