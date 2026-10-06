import gurobipy as gp
import pandas as pd
import numpy as np
import re

def solve_project_selection():
    df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/200pct/S1/Large-scale-or/Mixture_testing/Mixture4/project.csv', sep=',')
    project_ids = df['Project ID'].astype(int).tolist()
    n_projects = len(project_ids)
    if n_projects != 110:
        raise ValueError(f'Expected 110 projects, found {n_projects}')
    npv = dict(zip(df['Project ID'].astype(int), df['NPV (k$)'].astype(float)))
    capital = dict(zip(df['Project ID'].astype(int), df['Capital (k$)'].astype(float)))
    if set(project_ids) != set(npv.keys()) or set(project_ids) != set(capital.keys()):
        raise ValueError('Mismatch in project IDs for NPV or Capital columns.')

    def find_project_id_by_name(name):
        name_norm = re.sub('\\s+', ' ', name.strip()).casefold()
        matches = df['Project Name'].apply(lambda x: re.sub('\\s+', ' ', str(x).strip()).casefold() == name_norm)
        ids = df.loc[matches, 'Project ID'].astype(int).tolist()
        if len(ids) != 1:
            raise ValueError(f"Could not uniquely identify project '{name}'. Found: {ids}")
        return ids[0]
    pid_1 = find_project_id_by_name('Infrastructure Upgrade')
    pid_4 = find_project_id_by_name('R&D Initiative Alpha')
    pid_5 = find_project_id_by_name('Staff Training Program')
    pid_6 = find_project_id_by_name('System Automation')
    pid_7 = find_project_id_by_name('Global Expansion Pilot')
    pid_10 = find_project_id_by_name('Customer Experience Platform')
    m = gp.Model('ProjectSelection')
    y = m.addVars(project_ids, vtype=gp.GRB.BINARY, name='')
    m.setObjective(gp.quicksum((npv[i] * y[i] for i in project_ids)), gp.GRB.MAXIMIZE)
    m.addConstr(gp.quicksum((capital[i] * y[i] for i in project_ids)) <= 1000, name='budget')
    m.addConstr(y[pid_4] + y[pid_7] <= 1, name='mutual_excl_4_7')
    m.addConstr(y[pid_6] <= y[pid_1], name='prereq_6_1')
    m.addConstr(y[pid_10] <= y[pid_5], name='contingent_10_5')
    m.Params.MIPGap = 0.0001
    m.optimize()
    if m.status == gp.GRB.OPTIMAL:
        print(f'ObjVal: {m.objVal}')
        for i in project_ids:
            print(f'y[{i}] {y[i].VarName} {y[i].X}')
    else:
        print(f'Solver status: {m.status}')
    return m
m = solve_project_selection()