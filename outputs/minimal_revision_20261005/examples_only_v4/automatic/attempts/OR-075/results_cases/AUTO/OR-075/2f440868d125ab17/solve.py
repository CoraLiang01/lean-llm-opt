import gurobipy as gp
import pandas as pd
import numpy as np
project_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture4/project.csv'
df = pd.read_csv(project_path, sep=',')
if df['Project ID'].isnull().any():
    raise ValueError('Missing Project ID in project.csv')
project_ids = df['Project ID'].astype(int).tolist()
n_projects = len(project_ids)
if n_projects != 110:
    raise ValueError(f'Expected 110 projects, found {n_projects}')
capital = dict(zip(df['Project ID'].astype(int), df['Capital (k$)'].astype(float)))
npv = dict(zip(df['Project ID'].astype(int), df['NPV (k$)'].astype(float)))
if set(capital.keys()) != set(project_ids) or set(npv.keys()) != set(project_ids):
    raise ValueError('Mismatch in project IDs for capital or NPV')
name_to_id = dict(zip(df['Project Name'].str.casefold().str.strip(), df['Project ID'].astype(int)))

def get_pid(name):
    key = name.casefold().strip()
    if key not in name_to_id:
        raise ValueError(f"Project name '{name}' not found in project.csv")
    return int(name_to_id[key])
pid_1 = get_pid('Infrastructure Upgrade')
pid_4 = get_pid('R&D Initiative Alpha')
pid_5 = get_pid('Staff Training Program')
pid_6 = get_pid('System Automation')
pid_7 = get_pid('Global Expansion Pilot')
pid_10 = get_pid('Customer Experience Platform')
m = gp.Model('ProjectSelection')
x = m.addVars(project_ids, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((npv[i] * x[i] for i in project_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((capital[i] * x[i] for i in project_ids)) <= 1000, name='budget')
m.addConstr(x[pid_4] + x[pid_7] <= 1, name='mutual_excl_4_7')
m.addConstr(x[pid_6] <= x[pid_1], name='prereq_6_1')
m.addConstr(x[pid_10] <= x[pid_5], name='contingent_10_5')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.Status}')