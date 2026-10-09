import gurobipy as gp
import pandas as pd
import numpy as np
import re

def parse_predecessors(s):
    if pd.isna(s) or str(s).strip() == '':
        return []
    return [x.strip() for x in str(s).split(';') if x.strip() != '']
activities_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant13/inputs/project_activities.csv', dtype=str, keep_default_na=False)
params_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant13/inputs/project_parameters.csv', dtype=str, keep_default_na=False)
activity_ids = activities_df['Activity'].tolist()

def to_int_series(df, col):
    return df[col].astype(int)
normal_duration = dict(zip(activities_df['Activity'], to_int_series(activities_df, 'NormalDuration')))
crash_duration = dict(zip(activities_df['Activity'], to_int_series(activities_df, 'CrashDuration')))
crash_cost_per_day = dict(zip(activities_df['Activity'], to_int_series(activities_df, 'CrashCostPerDay')))
crash_lb = {aid: 0 for aid in activity_ids}
crash_ub = {aid: normal_duration[aid] - crash_duration[aid] for aid in activity_ids}
predecessors = {}
for (idx, row) in activities_df.iterrows():
    aid = row['Activity']
    preds = parse_predecessors(row['Predecessors'])
    predecessors[aid] = preds
deadline_row = params_df[params_df['Parameter'].str.strip().str.casefold() == 'projectdeadline']
if deadline_row.empty:
    raise ValueError('ProjectDeadline parameter not found in project_parameters.csv')
project_deadline = int(deadline_row.iloc[0]['Value'])
m = gp.Model('ProjectCrashing')
s_vars = m.addVars(activity_ids, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
z_vars = m.addVars(activity_ids, lb=0, ub=[crash_ub[aid] for aid in activity_ids], vtype=gp.GRB.INTEGER, name='')
T_var = m.addVar(lb=0.0, vtype=gp.GRB.CONTINUOUS, name='T')
m.setObjective(gp.quicksum((crash_cost_per_day[aid] * z_vars[aid] for aid in activity_ids)), gp.GRB.MINIMIZE)
for i in activity_ids:
    for j in predecessors[i]:
        if j not in activity_ids:
            raise ValueError(f"Predecessor '{j}' of activity '{i}' not found in activity list.")
        m.addConstr(s_vars[i] >= s_vars[j] + (normal_duration[j] - z_vars[j]), name=f'prec_{j}_to_{i}')
for i in activity_ids:
    m.addConstr(z_vars[i] >= crash_lb[i], name=f'z_lb_{i}')
    m.addConstr(z_vars[i] <= crash_ub[i], name=f'z_ub_{i}')
for i in activity_ids:
    m.addConstr(T_var >= s_vars[i] + (normal_duration[i] - z_vars[i]), name=f'T_ge_finish_{i}')
m.addConstr(T_var <= project_deadline, name='project_deadline')
m.optimize()