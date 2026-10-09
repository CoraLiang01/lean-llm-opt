import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/200pct/S2/Large-scale-or/Mixture_testing/Mixture11/courses_42.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
discipline_col = 'discipline'
or_mask = df[discipline_col].str.strip().str.casefold() == 'operations research'
or_courses_df = df[or_mask].copy()
if or_courses_df.empty:
    raise ValueError("No courses found with discipline 'Operations Research'.")
course_id_col = 'course_id'
credits_col = 'credits'
interest_points_col = 'interest_points'
for col in [course_id_col, credits_col, interest_points_col]:
    if col not in or_courses_df.columns:
        raise KeyError(f"Required column '{col}' not found in CSV.")
or_course_ids = or_courses_df[course_id_col].tolist()
credits = {}
interest_points = {}
for (idx, row) in or_courses_df.iterrows():
    cid = row[course_id_col]
    try:
        credits[cid] = int(row[credits_col])
    except Exception:
        raise ValueError(f'Invalid credits value for course_id {cid}: {row[credits_col]}')
    try:
        interest_points[cid] = int(row[interest_points_col])
    except Exception:
        raise ValueError(f'Invalid interest_points value for course_id {cid}: {row[interest_points_col]}')
m = gp.Model('OR_Course_Selection')
x_vars = m.addVars(or_course_ids, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((interest_points[cid] * x_vars[cid] for cid in or_course_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((credits[cid] * x_vars[cid] for cid in or_course_ids)) <= 20, name='credit_limit')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    selected = [cid for cid in or_course_ids if x_vars[cid].X > 0.5]
    total_credits = sum((credits[cid] for cid in selected))
    total_interest = sum((interest_points[cid] for cid in selected))
    print(f'Optimal total value/cost: {m.objVal:.0f} (Total Interest Points)')
    print(f'Total credits used: {total_credits} / 20')
    print('--- Selected Operations Research Courses ---')
    for cid in selected:
        cname = or_courses_df.loc[or_courses_df[course_id_col] == cid, 'course_name'].values[0]
        print(f'  {cid}: {cname} | Credits: {credits[cid]}, Interest Points: {interest_points[cid]}')
else:
    print(f'No optimal solution found. Status: {m.status}')