import gurobipy as gp
import pandas as pd
import numpy as np
option_catalog_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant6/inputs/option_catalog.csv'
resource_limits_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant6/inputs/resource_limits.csv'
option_df = pd.read_csv(option_catalog_path, sep=',')
limits_df = pd.read_csv(resource_limits_path, sep=',')
families = sorted(option_df['Family'].astype(str).unique())
options = sorted(option_df['Option'].astype(str).unique())
family_options = option_df.groupby('Family')['Option'].apply(lambda s: sorted(s.astype(str).unique())).to_dict()

def build_param_dict(col):
    return {(str(row['Family']), str(row['Option'])): int(row[col]) for (_, row) in option_df.iterrows()}
value = build_param_dict('Value')
weight = build_param_dict('Weight')
budgetuse = build_param_dict('BudgetUse')
limits = {str(row['Resource']).strip(): int(row['Limit']) for (_, row) in limits_df.iterrows()}
for res in ['Weight', 'BudgetUse']:
    if res not in limits:
        raise ValueError(f"Resource limit for '{res}' not found in resource_limits.csv.")
m = gp.Model('MultiChoiceKnapsack')
x = {}
for f in families:
    for o in family_options[f]:
        x[f, o] = m.addVar(vtype=gp.GRB.BINARY, name='x')
m.setObjective(gp.quicksum((value[f, o] * x[f, o] for f in families for o in family_options[f])), gp.GRB.MAXIMIZE)
for f in families:
    m.addConstr(gp.quicksum((x[f, o] for o in family_options[f])) == 1, name=f'family_{f}_select')
m.addConstr(gp.quicksum((weight[f, o] * x[f, o] for f in families for o in family_options[f])) <= limits['Weight'], name='weight_limit')
m.addConstr(gp.quicksum((budgetuse[f, o] * x[f, o] for f in families for o in family_options[f])) <= limits['BudgetUse'], name='budget_limit')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f}')
    print('--- Bundle Selection ---')
    for f in families:
        for o in family_options[f]:
            if x[f, o].X > 0.5:
                print(f'Family {f}: Option {o} (Value={value[f, o]}, Weight={weight[f, o]}, BudgetUse={budgetuse[f, o]})')
    total_weight = sum((weight[f, o] * x[f, o].X for f in families for o in family_options[f]))
    total_budget = sum((budgetuse[f, o] * x[f, o].X for f in families for o in family_options[f]))
    print(f"Total Weight: {total_weight:.0f} / {limits['Weight']}")
    print(f"Total BudgetUse: {total_budget:.0f} / {limits['BudgetUse']}")
else:
    print(f'No optimal solution found. Status: {m.status}')