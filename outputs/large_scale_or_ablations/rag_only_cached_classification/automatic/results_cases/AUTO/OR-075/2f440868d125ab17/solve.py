import pandas as pd
import numpy as np
from gurobipy import Model, GRB
project_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture4/project.csv', sep=',')
project_df['Project ID'] = project_df['Project ID'].astype(int)
project_ids = project_df['Project ID'].tolist()
capital = dict(zip(project_df['Project ID'], project_df['Capital (k$)']))
npv = dict(zip(project_df['Project ID'], project_df['NPV (k$)']))
required_ids = [1, 4, 5, 6, 7, 10]
missing_ids = [pid for pid in required_ids if pid not in project_ids]
if missing_ids:
    raise ValueError(f'Required Project IDs missing from data: {missing_ids}')
m = Model('ProjectSelection')
x = m.addVars(project_ids, vtype=GRB.BINARY, lb=0, ub=1, name='')
m.setObjective(sum((npv[i] * x[i] for i in project_ids)), GRB.MAXIMIZE)
m.addConstr(sum((capital[i] * x[i] for i in project_ids)) <= 1000, name='budget')
m.addConstr(x[4] + x[7] <= 1, name='mutual_excl_4_7')
m.addConstr(x[6] <= x[1], name='prereq_6_1')
m.addConstr(x[10] <= x[5], name='contingent_10_5')
m.optimize()