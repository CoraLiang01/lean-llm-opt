import gurobipy as gp
import pandas as pd
import numpy as np
activities_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant21/inputs/project_activities.csv', sep=',')
params_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant21/inputs/project_parameters.csv', sep=',')
activities_df['Activity'] = activities_df['Activity'].astype(str).str.strip()
activities_df['Predecessors'] = activities_df['Predecessors'].astype(str).str.strip()
activities = list(activities_df['Activity'])
crash_cost = {}
normal_duration = {}
crash_duration = {}
max_crash = {}
for (_, row) in activities_df.iterrows():
    act = row['Activity']
    crash_cost[act] = int(row['CrashCostPerDay'])
    normal_duration[act] = int(row['NormalDuration'])
    crash_duration[act] = int(row['CrashDuration'])
    max_crash[act] = normal_duration[act] - crash_duration[act]
    if max_crash[act] < 0:
        raise ValueError(f'Activity {act} has CrashDuration > NormalDuration.')
predecessors = {}
for (_, row) in activities_df.iterrows():
    act = row['Activity']
    preds_raw = row['Predecessors']
    if preds_raw == '' or preds_raw.lower() == 'nan':
        predecessors[act] = []
    else:
        preds = [p.strip() for p in preds_raw.split(';') if p.strip() != '']
        predecessors[act] = preds
deadline_row = params_df[params_df['Parameter'].str.casefold() == 'projectdeadline']
if deadline_row.empty:
    raise ValueError('ProjectDeadline parameter not found in project_parameters.csv')
project_deadline = int(deadline_row['Value'].iloc[0])
m = gp.Model('ProjectCrashing')
s = m.addVars(activities, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
z = m.addVars(activities, lb=0, ub=[max_crash[a] for a in activities], vtype=gp.GRB.INTEGER, name='')
T = m.addVar(lb=0.0, vtype=gp.GRB.CONTINUOUS, name='T')
m.setObjective(gp.quicksum((crash_cost[a] * z[a] for a in activities)), gp.GRB.MINIMIZE)
for i in activities:
    for j in predecessors[i]:
        if j not in activities:
            raise ValueError(f'Predecessor {j} of activity {i} not found in activity list.')
        m.addConstr(s[i] >= s[j] + (normal_duration[j] - z[j]), name=f'prec_{j}_to_{i}')
for i in activities:
    m.addConstr(z[i] >= 0, name=f'z_lb_{i}')
    m.addConstr(z[i] <= max_crash[i], name=f'z_ub_{i}')
for i in activities:
    m.addConstr(T >= s[i] + (normal_duration[i] - z[i]), name=f'T_ge_finish_{i}')
m.addConstr(T <= project_deadline, name='deadline')
m.Params.MIPGap = 0.0001
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')