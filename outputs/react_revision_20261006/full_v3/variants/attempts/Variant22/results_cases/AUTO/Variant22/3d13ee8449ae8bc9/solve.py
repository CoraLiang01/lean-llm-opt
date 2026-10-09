import gurobipy as gp
import pandas as pd
import numpy as np
option_catalog_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant22/inputs/option_catalog.csv'
resource_limits_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant22/inputs/resource_limits.csv'
option_df = pd.read_csv(option_catalog_path, sep=',')
option_df['Family'] = option_df['Family'].astype(str).str.strip()
option_df['Option'] = option_df['Option'].astype(str).str.strip()
limits_df = pd.read_csv(resource_limits_path, sep=',')
limits_df['Resource'] = limits_df['Resource'].astype(str).str.strip()
families = sorted(option_df['Family'].unique())
options_by_family = {fam: sorted(option_df.loc[option_df['Family'] == fam, 'Option'].unique()) for fam in families}
family_option_pairs = [(fam, opt) for fam in families for opt in options_by_family[fam]]
resources = sorted(limits_df['Resource'].unique())
resource_limits = {row['Resource']: int(row['Limit']) for (_, row) in limits_df.iterrows()}
value = {}
weight = {}
labor = {}
for (_, row) in option_df.iterrows():
    fam = str(row['Family']).strip()
    opt = str(row['Option']).strip()
    value[fam, opt] = int(row['Value'])
    weight[fam, opt] = int(row['Weight'])
    labor[fam, opt] = int(row['LaborHours'])
for (fam, opt) in family_option_pairs:
    if (fam, opt) not in value or (fam, opt) not in weight or (fam, opt) not in labor:
        raise ValueError(f'Missing coefficients for family {fam}, option {opt}')
m = gp.Model('multi_choice_knapsack')
x = m.addVars(family_option_pairs, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((value[fam, opt] * x[fam, opt] for (fam, opt) in family_option_pairs)), gp.GRB.MAXIMIZE)
for fam in families:
    m.addConstr(gp.quicksum((x[fam, opt] for opt in options_by_family[fam])) == 1, name=f'one_option_{fam}')
if 'Weight' not in resource_limits:
    raise ValueError("Missing 'Weight' resource limit in resource_limits.csv")
m.addConstr(gp.quicksum((weight[fam, opt] * x[fam, opt] for (fam, opt) in family_option_pairs)) <= resource_limits['Weight'], name='weight_limit')
if 'LaborHours' not in resource_limits:
    raise ValueError("Missing 'LaborHours' resource limit in resource_limits.csv")
m.addConstr(gp.quicksum((labor[fam, opt] * x[fam, opt] for (fam, opt) in family_option_pairs)) <= resource_limits['LaborHours'], name='labor_limit')
m.setParam('MIPGap', 0.0001)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for (fam, opt) in family_option_pairs:
        var = x[fam, opt]
        print(f'{var.VarName} {var.X}')
else:
    print(f'Solver status: {m.status}')