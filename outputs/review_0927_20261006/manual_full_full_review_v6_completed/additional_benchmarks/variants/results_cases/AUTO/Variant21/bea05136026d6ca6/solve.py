import gurobipy as gp
import pandas as pd
import numpy as np
import re
activities_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant21/inputs/project_activities.csv', sep=',', dtype=str, keep_default_na=False)
parameters_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant21/inputs/project_parameters.csv', sep=',', dtype=str, keep_default_na=False)
activity_ids = activities_df['Activity'].astype(str).str.strip()
activity_set = list(activity_ids)

def parse_preds(preds):
    preds = preds.strip()
    if preds == '':
        return []
    return [p.strip() for p in re.split('[;,]', preds) if p.strip() != '']
predecessors_dict = {}
for idx, row in activities_df.iterrows():
    act = row['Activity'].strip()
    preds = parse_preds(row['Predecessors'])
    predecessors_dict[act] = preds
normal_duration = {}
crash_duration = {}
crash_cost_per_day = {}
for idx, row in activities_df.iterrows():
    act = row['Activity'].strip()
    try:
        nd = int(row['NormalDuration'])
        cd = int(row['CrashDuration'])
        cc = int(row['CrashCostPerDay'])
    except Exception as e:
        raise ValueError(f'Invalid numeric value for activity {act}: {e}')
    normal_duration[act] = nd
    crash_duration[act] = cd
    crash_cost_per_day[act] = cc
crash_bounds = {act: (0, normal_duration[act] - crash_duration[act]) for act in activity_set}
deadline_row = parameters_df[parameters_df['Parameter'].str.strip().str.casefold() == 'projectdeadline']
if deadline_row.empty:
    raise ValueError('ProjectDeadline parameter not found in project_parameters.csv')
project_deadline = int(deadline_row.iloc[0]['Value'])
m = gp.Model('ProjectCrashing')
s_vars = m.addVars(activity_set, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
z_vars = m.addVars(activity_set, lb=0, ub=[crash_bounds[act][1] for act in activity_set], vtype=gp.GRB.INTEGER, name='')
T_var = m.addVar(lb=0.0, vtype=gp.GRB.CONTINUOUS, name='T')
m.setObjective(gp.quicksum((crash_cost_per_day[act] * z_vars[act] for act in activity_set)), gp.GRB.MINIMIZE)
for act in activity_set:
    for pred in predecessors_dict[act]:
        if pred not in activity_set:
            raise ValueError(f"Predecessor '{pred}' of activity '{act}' not found in activity list.")
        m.addConstr(s_vars[act] >= s_vars[pred] + (normal_duration[pred] - z_vars[pred]), name=f'prec_{pred}_to_{act}')
for act in activity_set:
    m.addConstr(s_vars[act] + (normal_duration[act] - z_vars[act]) <= T_var, name=f'completion_{act}')
m.addConstr(T_var <= project_deadline, name='project_deadline')
m.optimize()