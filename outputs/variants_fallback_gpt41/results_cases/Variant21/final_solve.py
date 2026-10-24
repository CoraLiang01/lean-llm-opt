import gurobipy as gp
import pandas as pd
import numpy as np
import re
activities_path = 'benchmark_dataset/Variant21/inputs/project_activities.csv'
activities_df = pd.read_csv(activities_path, sep=',', dtype=str, keep_default_na=False)
activities_df['Activity'] = activities_df['Activity'].str.strip()
activity_ids = activities_df['Activity'].tolist()

def to_int_series(df, col):
    return df[col].astype(int)
normal_duration = dict(zip(activities_df['Activity'], to_int_series(activities_df, 'NormalDuration')))
crash_duration = dict(zip(activities_df['Activity'], to_int_series(activities_df, 'CrashDuration')))
crash_cost_per_day = dict(zip(activities_df['Activity'], to_int_series(activities_df, 'CrashCostPerDay')))
crash_bounds = {i: (0, normal_duration[i] - crash_duration[i]) for i in activity_ids}

def parse_preds(s):
    s = s.strip()
    if s == '':
        return []
    return [x.strip() for x in re.split('[;,\\s]+', s) if x.strip()]
predecessors = {}
for (idx, row) in activities_df.iterrows():
    act = row['Activity']
    preds = parse_preds(row['Predecessors'])
    predecessors[act] = preds
params_path = 'benchmark_dataset/Variant21/inputs/project_parameters.csv'
params_df = pd.read_csv(params_path, sep=',', dtype=str, keep_default_na=False)
params_df['Parameter'] = params_df['Parameter'].str.strip().str.casefold()
params_df['Value'] = params_df['Value'].astype(int)
deadline_row = params_df[params_df['Parameter'] == 'projectdeadline']
if len(deadline_row) != 1:
    raise ValueError('ProjectDeadline parameter not found or not unique in project_parameters.csv')
project_deadline = int(deadline_row['Value'].iloc[0])
m = gp.Model('ProjectCrashing')
s_vars = m.addVars(activity_ids, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
z_vars = m.addVars(activity_ids, lb={i: crash_bounds[i][0] for i in activity_ids}, ub={i: crash_bounds[i][1] for i in activity_ids}, vtype=gp.GRB.INTEGER, name='')
T_var = m.addVar(lb=0.0, vtype=gp.GRB.CONTINUOUS, name='T')
m.setObjective(gp.quicksum((crash_cost_per_day[i] * z_vars[i] for i in activity_ids)), gp.GRB.MINIMIZE)
for i in activity_ids:
    for j in predecessors[i]:
        if j not in activity_ids:
            raise ValueError(f"Predecessor '{j}' of activity '{i}' not found in activity list.")
        m.addConstr(s_vars[i] >= s_vars[j] + (normal_duration[j] - z_vars[j]), name=f'prec_{i}_{j}')
for i in activity_ids:
    m.addConstr(T_var >= s_vars[i] + (normal_duration[i] - z_vars[i]), name=f'compl_{i}')
m.addConstr(T_var <= project_deadline, name='deadline')
m.optimize()