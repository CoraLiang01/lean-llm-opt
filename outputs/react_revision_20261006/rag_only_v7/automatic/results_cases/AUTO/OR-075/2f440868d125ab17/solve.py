import pandas as pd
import gurobipy as gp
from gurobipy import GRB
project_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture4/project.csv'
df = pd.read_csv(project_path, sep=',', dtype=str, keep_default_na=False)
for col in ['Project ID', 'Capital (k$)', 'NPV (k$)']:
    df[col] = df[col].astype(int)
project_ids = df['Project ID'].tolist()
npv_dict = dict(zip(df['Project ID'], df['NPV (k$)']))
capital_dict = dict(zip(df['Project ID'], df['Capital (k$)']))
name_to_id = {name.casefold().strip(): pid for (name, pid) in zip(df['Project Name'], df['Project ID'])}
required_projects = [('r&d initiative alpha', 'Project 4'), ('global expansion pilot', 'Project 7'), ('system automation', 'Project 6'), ('infrastructure upgrade', 'Project 1'), ('customer experience platform', 'Project 10'), ('staff training program', 'Project 5')]
for (pname, desc) in required_projects:
    if pname not in name_to_id:
        raise ValueError(f"Required project '{desc}' with name '{pname}' not found in data.")
proj_4 = name_to_id['r&d initiative alpha']
proj_7 = name_to_id['global expansion pilot']
proj_6 = name_to_id['system automation']
proj_1 = name_to_id['infrastructure upgrade']
proj_10 = name_to_id['customer experience platform']
proj_5 = name_to_id['staff training program']
m = gp.Model('project_selection')
x_vars = m.addVars(project_ids, vtype=GRB.BINARY, lb=0, ub=1, name='')
m.setObjective(gp.quicksum((x_vars[i] * npv_dict[i] for i in project_ids)), GRB.MAXIMIZE)
m.addConstr(gp.quicksum((x_vars[i] * capital_dict[i] for i in project_ids)) <= 1000, name='budget')
m.addConstr(x_vars[proj_4] + x_vars[proj_7] <= 1, name='mutual_excl_4_7')
m.addConstr(x_vars[proj_6] <= x_vars[proj_1], name='prereq_6_1')
m.addConstr(x_vars[proj_10] <= x_vars[proj_5], name='contingent_10_5')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for i in project_ids:
        var = x_vars[i]
        print(f'{var.VarName} {var.X}')
else:
    print(f'Solver status: {m.Status}')