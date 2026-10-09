import gurobipy as gp
import pandas as pd
import numpy as np
import re
activities_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant3/inputs/project_activities.csv', sep=',', dtype=str, keep_default_na=False)
parameters_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant3/inputs/project_parameters.csv', sep=',', dtype=str, keep_default_na=False)
activities = activities_df['Activity'].astype(str).str.strip()
activity_list = activities.tolist()
normal_duration = {}
crash_duration = {}
crash_cost_per_day = {}
crash_max = {}
for (idx, row) in activities_df.iterrows():
    act = str(row['Activity']).strip()
    try:
        nd = int(row['NormalDuration'])
        cd = int(row['CrashDuration'])
        cc = int(row['CrashCostPerDay'])
    except Exception as e:
        raise ValueError(f"Non-integer duration or cost for activity '{act}': {e}")
    normal_duration[act] = nd
    crash_duration[act] = cd
    crash_cost_per_day[act] = cc
    crash_max[act] = nd - cd
    if crash_max[act] < 0:
        raise ValueError(f"Crash duration exceeds normal duration for activity '{act}'.")
predecessors = {}
for (idx, row) in activities_df.iterrows():
    act = str(row['Activity']).strip()
    preds_raw = str(row['Predecessors']).strip()
    if preds_raw == '' or preds_raw.lower() in ['nan', 'none']:
        predecessors[act] = []
    else:
        preds = [p.strip() for p in re.split(';', preds_raw) if p.strip() != '']
        predecessors[act] = preds
deadline_row = parameters_df[parameters_df['Parameter'].str.strip().str.casefold() == 'projectdeadline']
if deadline_row.empty:
    raise ValueError('ProjectDeadline parameter not found in project_parameters.csv')
try:
    project_deadline = int(deadline_row.iloc[0]['Value'])
except Exception as e:
    raise ValueError(f'ProjectDeadline value is not an integer: {e}')
m = gp.Model('ProjectCrashing')
s_vars = m.addVars(activity_list, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
z_vars = m.addVars(activity_list, lb=0, vtype=gp.GRB.INTEGER, name='')
T_var = m.addVar(lb=0.0, vtype=gp.GRB.CONTINUOUS, name='T')
m.setObjective(gp.quicksum((crash_cost_per_day[act] * z_vars[act] for act in activity_list)), gp.GRB.MINIMIZE)
for i in activity_list:
    for j in predecessors[i]:
        if j not in activity_list:
            raise ValueError(f"Predecessor '{j}' of activity '{i}' not found in activity list.")
        m.addConstr(s_vars[i] >= s_vars[j] + (normal_duration[j] - z_vars[j]), name=f'prec_{j}_to_{i}')
for i in activity_list:
    m.addConstr(z_vars[i] >= 0, name=f'z_lb_{i}')
    m.addConstr(z_vars[i] <= crash_max[i], name=f'z_ub_{i}')
for i in activity_list:
    m.addConstr(T_var >= s_vars[i] + (normal_duration[i] - z_vars[i]), name=f'T_ge_finish_{i}')
m.addConstr(T_var <= project_deadline, name='project_deadline')
m.optimize()