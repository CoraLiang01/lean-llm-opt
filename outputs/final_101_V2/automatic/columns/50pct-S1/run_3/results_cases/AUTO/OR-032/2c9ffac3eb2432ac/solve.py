import gurobipy as gp
import pandas as pd
import numpy as np
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete 20260928/50pct/S1/Large-scale-or/Mixture_testing/Mixture11/courses_42.csv'
df = pd.read_csv(csv_path, sep=',')
or_mask = df['discipline'].astype(str).str.casefold().str.strip() == 'operations research'
or_courses_df = df[or_mask].copy()
if or_courses_df.empty:
    raise ValueError("No courses found in the 'Operations Research' discipline.")
OR_COURSES = list(or_courses_df['course_id'])
credits = or_courses_df.set_index('course_id')['credits'].to_dict()
interest_points = or_courses_df.set_index('course_id')['interest_points'].to_dict()
for cid in OR_COURSES:
    if cid not in credits or cid not in interest_points:
        raise ValueError(f'Missing credits or interest_points for course_id {cid}')
m = gp.Model('OR_Course_Selection')
x = m.addVars(OR_COURSES, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((interest_points[cid] * x[cid] for cid in OR_COURSES)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((credits[cid] * x[cid] for cid in OR_COURSES)) <= 20, name='credit_limit')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    selected = [cid for cid in OR_COURSES if x[cid].X > 0.5]
    total_credits = sum((credits[cid] for cid in selected))
    total_interest = sum((interest_points[cid] for cid in selected))
    print(f'Optimal total interest points: {total_interest}')
    print(f'Total credits used: {total_credits}')
    print('\nSelected Operations Research courses:')
    for cid in selected:
        row = or_courses_df[or_courses_df['course_id'] == cid].iloc[0]
        print(f"  {cid}: {row['course_name']} (Credits: {row['credits']}, Interest Points: {row['interest_points']})")
else:
    print(f'No optimal solution found. Status: {m.status}')