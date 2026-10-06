import gurobipy as gp
import pandas as pd
import numpy as np
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete 20260928/50pct/S3/Large-scale-or/Mixture_testing/Mixture11/courses_42.csv'
df = pd.read_csv(csv_path, sep=',')
discipline_col = 'discipline'
or_mask = df[discipline_col].astype(str).str.casefold().str.strip() == 'operations research'
or_courses_df = df[or_mask].copy()
or_course_ids = list(or_courses_df['course_id'])
if len(or_course_ids) == 0:
    raise ValueError("No courses found in the 'Operations Research' discipline.")
credits = or_courses_df.set_index('course_id')['credits'].to_dict()
interest_points = or_courses_df.set_index('course_id')['interest_points'].to_dict()
for cid in or_course_ids:
    if cid not in credits or cid not in interest_points:
        raise ValueError(f'Missing data for course_id {cid}')
m = gp.Model('OR_Course_Selection')
x = m.addVars(or_course_ids, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((interest_points[cid] * x[cid] for cid in or_course_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((credits[cid] * x[cid] for cid in or_course_ids)) <= 20, name='credit_limit')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    selected = [cid for cid in or_course_ids if x[cid].X > 0.5]
    total_credits = sum((credits[cid] for cid in selected))
    total_interest = sum((interest_points[cid] for cid in selected))
    print(f'Optimal total value/cost: {m.objVal:.0f} (Total Interest Points)')
    print(f'Total credits used: {total_credits} / 20')
    print('\n--- Selected Operations Research Courses ---')
    for cid in selected:
        cname = or_courses_df.loc[or_courses_df['course_id'] == cid, 'course_name'].values[0]
        print(f'  {cid}: {cname} | Credits: {credits[cid]}, Interest Points: {interest_points[cid]}')
else:
    print(f'No optimal solution found. Status: {m.status}')