import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/50pct/S3/Large-scale-or/Mixture_testing/Mixture11/courses_42.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
discipline_col = 'discipline'
or_mask = df[discipline_col].str.casefold().str.strip() == 'operations research'
df_or = df[or_mask].copy()
or_course_ids = df_or['course_id'].tolist()
if df_or['credits'].isnull().any() or df_or['interest_points'].isnull().any():
    raise ValueError("Missing values in 'credits' or 'interest_points' for Operations Research courses.")
credits = df_or.set_index('course_id')['credits'].astype(int).to_dict()
interest_points = df_or.set_index('course_id')['interest_points'].astype(int).to_dict()
m = gp.Model('OR_Course_Selection')
x_vars = m.addVars(or_course_ids, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((interest_points[cid] * x_vars[cid] for cid in or_course_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((credits[cid] * x_vars[cid] for cid in or_course_ids)) <= 20, name='credit_limit')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    selected = [cid for cid in or_course_ids if x_vars[cid].X > 0.5]
    total_credits = sum((credits[cid] for cid in selected))
    total_interest = sum((interest_points[cid] for cid in selected))
    print(f'Optimal total interest points: {m.objVal}')
    print(f'Total credits used: {total_credits}')
    print('Selected Operations Research courses:')
    for cid in selected:
        cname = df_or.loc[df_or['course_id'] == cid, 'course_name'].values[0]
        print(f'  {cid}: {cname} (Credits: {credits[cid]}, Interest Points: {interest_points[cid]})')
else:
    print(f'No optimal solution found. Status: {m.status}')