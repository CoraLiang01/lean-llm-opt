import pandas as pd
import gurobipy as gp
from gurobipy import GRB
project_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture4/project.csv'
df = pd.read_csv(project_path, sep=',')
if df['Project ID'].duplicated().any():
    raise ValueError('Duplicate Project IDs found in project.csv')
if df['Project ID'].isnull().any():
    raise ValueError('Missing Project IDs in project.csv')
project_ids = df['Project ID'].astype(int).tolist()
project_id_set = set(project_ids)
if len(project_ids) != 110:
    raise ValueError(f'Expected 110 projects, found {len(project_ids)}')
id_to_name = dict(zip(df['Project ID'].astype(int), df['Project Name'].astype(str)))
name_to_id = {v.casefold().strip(): k for k, v in id_to_name.items()}
capital = df.set_index('Project ID')['Capital (k$)'].astype(int).to_dict()
npv = df.set_index('Project ID')['NPV (k$)'].astype(int).to_dict()

def get_id_by_name(target_name):
    key = target_name.casefold().strip()
    if key not in name_to_id:
        raise ValueError(f"Project '{target_name}' not found in project.csv")
    return int(name_to_id[key])
proj_1 = get_id_by_name('Infrastructure Upgrade')
proj_4 = get_id_by_name('R&D Initiative Alpha')
proj_5 = get_id_by_name('Staff Training Program')
proj_6 = get_id_by_name('System Automation')
proj_7 = get_id_by_name('Global Expansion Pilot')
proj_10 = get_id_by_name('Customer Experience Platform')

def solve_problem():
    m = gp.Model('project_selection')
    x = m.addVars(project_ids, vtype=GRB.BINARY, name='')
    m.setObjective(gp.quicksum((npv[i] * x[i] for i in project_ids)), GRB.MAXIMIZE)
    m.addConstr(gp.quicksum((capital[i] * x[i] for i in project_ids)) <= 1000, name='budget')
    m.addConstr(x[proj_4] + x[proj_7] <= 1, name='mutually_exclusive_4_7')
    m.addConstr(x[proj_6] <= x[proj_1], name='prereq_6_requires_1')
    m.addConstr(x[proj_10] <= x[proj_5], name='contingent_10_requires_5')
    m.optimize()
    return m
m = solve_problem()