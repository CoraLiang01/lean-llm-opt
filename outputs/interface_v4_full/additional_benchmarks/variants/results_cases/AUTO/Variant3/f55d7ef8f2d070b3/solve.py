import gurobipy as gp
import pandas as pd
import numpy as np
import re
activities_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant3/inputs/project_activities.csv', sep=',')
params_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant3/inputs/project_parameters.csv', sep=',')
activities = activities_df['Activity'].astype(str).tolist()
NormalDuration = dict(zip(activities_df['Activity'].astype(str), activities_df['NormalDuration'].astype(int)))
CrashDuration = dict(zip(activities_df['Activity'].astype(str), activities_df['CrashDuration'].astype(int)))
CrashCostPerDay = dict(zip(activities_df['Activity'].astype(str), activities_df['CrashCostPerDay'].astype(int)))

def parse_preds(cell):
    if pd.isnull(cell) or str(cell).strip() == '':
        return []
    return [pred.strip() for pred in str(cell).split(';') if pred.strip() != '']
Predecessors = dict(zip(activities_df['Activity'].astype(str), activities_df['Predecessors'].apply(parse_preds)))
Successors = {a: [] for a in activities}
for i in activities:
    for pred in Predecessors[i]:
        if pred not in Successors:
            raise ValueError(f"Predecessor '{pred}' for activity '{i}' not found in activity list.")
        Successors[pred].append(i)
terminal_activities = [a for a in activities if len(Successors[a]) == 0]
deadline_row = params_df.loc[params_df['Parameter'].astype(str).str.casefold().str.strip() == 'projectdeadline']
if deadline_row.empty:
    raise ValueError('ProjectDeadline parameter not found in project_parameters.csv')
ProjectDeadline = int(deadline_row['Value'].iloc[0])
m = gp.Model('ProjectCrashing')
s = m.addVars(activities, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
z = m.addVars(activities, lb=0, vtype=gp.GRB.INTEGER, name='')
T = m.addVar(lb=0.0, vtype=gp.GRB.CONTINUOUS, name='T')
m.setObjective(gp.quicksum((CrashCostPerDay[i] * z[i] for i in activities)), gp.GRB.MINIMIZE)
for i in activities:
    for j in Predecessors[i]:
        m.addConstr(s[i] >= s[j] + (NormalDuration[j] - z[j]), name=f'prec_{j}_to_{i}')
for i in activities:
    max_crash = NormalDuration[i] - CrashDuration[i]
    m.addConstr(z[i] >= 0, name=f'z_lb_{i}')
    m.addConstr(z[i] <= max_crash, name=f'z_ub_{i}')
for i in terminal_activities:
    m.addConstr(T >= s[i] + (NormalDuration[i] - z[i]), name=f'T_terminal_{i}')
m.addConstr(T <= ProjectDeadline, name='T_deadline')
m.optimize()