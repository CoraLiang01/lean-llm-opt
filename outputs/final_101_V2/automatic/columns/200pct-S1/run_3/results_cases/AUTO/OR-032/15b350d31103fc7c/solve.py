import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete 20260928/200pct/S1/Large-scale-or/Mixture_testing/Mixture11/courses_42.csv'
df = pd.read_csv(csv_path, sep=',')
discipline_col = 'discipline'
ops_research_mask = df[discipline_col].astype(str).str.casefold().str.strip() == 'operations research'
df_or = df[ops_research_mask].copy()
if df_or.empty:
    raise ValueError("No courses found in 'Operations Research' discipline.")
course_ids = df_or['course_id'].astype(str).tolist()
credits = df_or.set_index('course_id')['credits'].astype(int).to_dict()
interest_points = df_or.set_index('course_id')['interest_points'].astype(int).to_dict()
for cid in course_ids:
    if cid not in credits or cid not in interest_points:
        raise ValueError(f'Missing data for course_id {cid}')
m = gp.Model('OR_Course_Selection')
x = m.addVars(course_ids, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((interest_points[c] * x[c] for c in course_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((credits[c] * x[c] for c in course_ids)) <= 20, name='credit_limit')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    total_interest = sum((interest_points[c] for c in course_ids if x[c].X > 0.5))
    total_credits = sum((credits[c] for c in course_ids if x[c].X > 0.5))
    print(f'Optimal total interest points: {total_interest}')
    print(f'Total credits used: {total_credits}')
    print('\nSelected Operations Research courses:')
    for c in course_ids:
        if x[c].X > 0.5:
            cname = df_or.loc[df_or['course_id'] == c, 'course_name'].values[0]
            print(f'  {c}: {cname} (Credits: {credits[c]}, Interest Points: {interest_points[c]})')
else:
    print(f'No optimal solution found. Status: {m.status}')