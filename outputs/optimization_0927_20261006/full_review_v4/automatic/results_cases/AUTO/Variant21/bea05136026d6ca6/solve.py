import gurobipy as gp
import pandas as pd
import numpy as np
import re

def parse_predecessors(s):
    s = s.strip()
    if not s:
        return []
    return [x.strip() for x in s.split(';') if x.strip()]
activities_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant21/inputs/project_activities.csv', dtype=str, keep_default_na=False)
parameters_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant21/inputs/project_parameters.csv', dtype=str, keep_default_na=False)
activity_ids = activities_df['Activity'].astype(str).str.strip()
if activity_ids.duplicated().any():
    raise ValueError('Duplicate Activity IDs found in project_activities.csv')
activity_ids = list(activity_ids)
normal_duration = {}
crash_duration = {}
crash_cost_per_day = {}
crash_day_max = {}
predecessors = {}
for (idx, row) in activities_df.iterrows():
    act = str(row['Activity']).strip()
    try:
        nd = int(row['NormalDuration'])
        cd = int(row['CrashDuration'])
        ccpd = int(row['CrashCostPerDay'])
    except Exception as e:
        raise ValueError(f'Non-integer duration or cost for activity {act}: {e}')
    if cd > nd:
        raise ValueError(f'CrashDuration > NormalDuration for activity {act}')
    normal_duration[act] = nd
    crash_duration[act] = cd
    crash_cost_per_day[act] = ccpd
    crash_day_max[act] = nd - cd
    predecessors[act] = parse_predecessors(row['Predecessors'])
deadline_row = parameters_df.loc[parameters_df['Parameter'].str.strip().str.casefold() == 'projectdeadline']
if deadline_row.empty:
    raise ValueError('ProjectDeadline not found in project_parameters.csv')
try:
    project_deadline = int(deadline_row.iloc[0]['Value'])
except Exception as e:
    raise ValueError(f'Non-integer ProjectDeadline: {e}')
m = gp.Model('ProjectCrashing')
s_vars = m.addVars(activity_ids, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
z_vars = m.addVars(activity_ids, lb=0, ub={act: crash_day_max[act] for act in activity_ids}, vtype=gp.GRB.INTEGER, name='')
T_var = m.addVar(lb=0.0, vtype=gp.GRB.CONTINUOUS, name='T')
m.setObjective(gp.quicksum((crash_cost_per_day[act] * z_vars[act] for act in activity_ids)), gp.GRB.MINIMIZE)
for i in activity_ids:
    for j in predecessors[i]:
        if j not in activity_ids:
            raise ValueError(f'Predecessor {j} of activity {i} not found in activity list')
        m.addConstr(s_vars[i] >= s_vars[j] + (normal_duration[j] - z_vars[j]), name=f'prec_{i}_after_{j}')
for i in activity_ids:
    m.addConstr(T_var >= s_vars[i] + (normal_duration[i] - z_vars[i]), name=f'T_ge_finish_{i}')
m.addConstr(T_var <= project_deadline, name='project_deadline')
m.optimize()