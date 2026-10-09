import gurobipy as gp
import pandas as pd
import numpy as np
import re
courses_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/50pct/S3/Large-scale-or/Mixture_testing/Mixture11/courses_42.csv'
df = pd.read_csv(courses_path, sep=',', dtype=str, keep_default_na=False)
discipline_col = 'discipline'
ops_research_mask = df[discipline_col].str.strip().str.casefold() == 'operations research'
ops_research_courses = df[ops_research_mask].copy()
if ops_research_courses.empty:
    raise ValueError("No courses found in the 'Operations Research' discipline.")
course_ids = ops_research_courses['course_id'].tolist()
interest_points = {}
credits = {}
for (idx, row) in ops_research_courses.iterrows():
    cid = row['course_id']
    try:
        interest_points[cid] = int(row['interest_points'])
        credits[cid] = int(row['credits'])
    except Exception as e:
        raise ValueError(f'Invalid numeric value for course_id {cid}: {e}')
m = gp.Model('OR_Course_Selection')
x_vars = m.addVars(course_ids, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((interest_points[cid] * x_vars[cid] for cid in course_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((credits[cid] * x_vars[cid] for cid in course_ids)) <= 20, name='credit_limit')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    total_interest = sum((interest_points[cid] for cid in course_ids if x_vars[cid].X > 0.5))
    total_credits = sum((credits[cid] for cid in course_ids if x_vars[cid].X > 0.5))
    print(f'Optimal total value/cost: {m.objVal:.0f} (Total Interest Points)')
    print(f'Total credits used: {total_credits} / 20')
    print('\n--- Selected Operations Research Courses ---')
    for cid in course_ids:
        if x_vars[cid].X > 0.5:
            row = ops_research_courses[ops_research_courses['course_id'] == cid].iloc[0]
            print(f"  {cid}: {row['course_name']} | Credits: {credits[cid]} | Interest Points: {interest_points[cid]}")
else:
    print(f'No optimal solution found. Status: {m.status}')