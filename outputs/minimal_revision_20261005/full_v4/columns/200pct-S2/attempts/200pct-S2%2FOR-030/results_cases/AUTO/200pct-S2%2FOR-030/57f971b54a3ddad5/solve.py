import gurobipy as gp
import pandas as pd
import numpy as np
import re

def solve_problem():
    path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/200pct/S2/Large-scale-or/Mixture_testing/Mixture4/project.csv'
    df = pd.read_csv(path, sep=',')
    if 'Project ID' not in df.columns:
        raise KeyError("Missing 'Project ID' column in project.csv")
    project_ids = df['Project ID'].astype(int).tolist()
    n_projects = len(project_ids)
    project_id_set = set(project_ids)
    if 'NPV (k$)' not in df.columns:
        raise KeyError("Missing 'NPV (k$)' column in project.csv")
    npv_dict = dict(zip(df['Project ID'].astype(int), df['NPV (k$)'].astype(float)))
    if set(npv_dict.keys()) != project_id_set:
        raise ValueError('Mismatch in Project IDs for NPV data')
    if 'Capital (k$)' not in df.columns:
        raise KeyError("Missing 'Capital (k$)' column in project.csv")
    capital_dict = dict(zip(df['Project ID'].astype(int), df['Capital (k$)'].astype(float)))
    if set(capital_dict.keys()) != project_id_set:
        raise ValueError('Mismatch in Project IDs for Capital data')
    if 'Project Name' not in df.columns:
        raise KeyError("Missing 'Project Name' column in project.csv")
    name_dict = dict(zip(df['Project ID'].astype(int), df['Project Name'].astype(str)))

    def find_project_id_by_name(target_name):
        target = re.sub('\\s+', ' ', target_name).strip().casefold()
        for (pid, pname) in name_dict.items():
            pname_norm = re.sub('\\s+', ' ', pname).strip().casefold()
            if pname_norm == target:
                return pid
        raise ValueError(f"Project with name '{target_name}' not found in project.csv")
    pid_4 = find_project_id_by_name('R&D Initiative Alpha')
    pid_7 = find_project_id_by_name('Global Expansion Pilot')
    pid_6 = find_project_id_by_name('System Automation')
    pid_1 = find_project_id_by_name('Infrastructure Upgrade')
    pid_10 = find_project_id_by_name('Customer Experience Platform')
    pid_5 = find_project_id_by_name('Staff Training Program')
    m = gp.Model('ProjectSelection')
    y = m.addVars(project_ids, vtype=gp.GRB.BINARY, name='')
    m.setObjective(gp.quicksum((npv_dict[i] * y[i] for i in project_ids)), gp.GRB.MAXIMIZE)
    m.addConstr(gp.quicksum((capital_dict[i] * y[i] for i in project_ids)) <= 1000, name='budget')
    m.addConstr(y[pid_4] + y[pid_7] <= 1, name='mutual_excl_4_7')
    m.addConstr(y[pid_6] <= y[pid_1], name='prereq_6_1')
    m.addConstr(y[pid_10] <= y[pid_5], name='contingent_10_5')
    m.setParam('MIPGap', 0.0001)
    m.optimize()
    if m.status == gp.GRB.OPTIMAL:
        print(f'ObjVal: {m.objVal}')
        for i in project_ids:
            print(f'y[{i}] {y[i].VarName} {y[i].X}')
    else:
        print(f'Solver status: {m.status}')
    return m
m = solve_problem()