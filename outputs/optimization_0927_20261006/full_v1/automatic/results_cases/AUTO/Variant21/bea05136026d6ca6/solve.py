import gurobipy as gp
import pandas as pd
import numpy as np
import re
activities_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant21/inputs/project_activities.csv', sep=',', dtype=str, keep_default_na=False)
parameters_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant21/inputs/project_parameters.csv', sep=',', dtype=str, keep_default_na=False)
activity_ids = activities_df['Activity'].tolist()
normal_duration = {}
crash_duration = {}
crash_cost_per_day = {}
crash_day_upper = {}
predecessors = {}
for (idx, row) in activities_df.iterrows():
    act = row['Activity']
    try:
        ndur = int(row['NormalDuration'])
        cdur = int(row['CrashDuration'])
        ccost = int(row['CrashCostPerDay'])
    except Exception as e:
        raise ValueError(f"Failed to convert numeric fields for activity '{act}': {e}")
    normal_duration[act] = ndur
    crash_duration[act] = cdur
    crash_cost_per_day[act] = ccost
    crash_day_upper[act] = ndur - cdur
    preds_raw = row['Predecessors'].strip()
    if preds_raw == '':
        preds = []
    else:
        preds = [p.strip() for p in preds_raw.split(';') if p.strip() != '']
    predecessors[act] = preds
deadline_row = parameters_df.loc[parameters_df['Parameter'].str.casefold().str.strip() == 'projectdeadline']
if deadline_row.empty:
    raise ValueError('ProjectDeadline parameter not found in project_parameters.csv')
try:
    project_deadline = int(deadline_row.iloc[0]['Value'])
except Exception as e:
    raise ValueError(f'Failed to convert ProjectDeadline value: {e}')
m = gp.Model('ProjectCrashing')
s_vars = m.addVars(activity_ids, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
z_vars = m.addVars(activity_ids, lb=0, ub=[crash_day_upper[act] for act in activity_ids], vtype=gp.GRB.INTEGER, name='')
T_var = m.addVar(lb=0.0, vtype=gp.GRB.CONTINUOUS, name='T')
m.setObjective(gp.quicksum((crash_cost_per_day[act] * z_vars[act] for act in activity_ids)), gp.GRB.MINIMIZE)
for i in activity_ids:
    for j in predecessors[i]:
        if j not in activity_ids:
            raise ValueError(f"Predecessor '{j}' of activity '{i}' not found in activity list.")
        m.addConstr(s_vars[i] >= s_vars[j] + (normal_duration[j] - z_vars[j]), name=f'prec_{j}_to_{i}')
for i in activity_ids:
    m.addConstr(T_var >= s_vars[i] + (normal_duration[i] - z_vars[i]), name=f'completion_{i}')
m.addConstr(T_var <= project_deadline, name='deadline')
m.optimize()