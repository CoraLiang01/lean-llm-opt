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
credits = {}
interest_points = {}
for (idx, row) in ops_research_courses.iterrows():
    cid = row['course_id']
    try:
        credits[cid] = int(row['credits'])
    except Exception as e:
        raise ValueError(f"Invalid credits value for course_id {cid}: {row['credits']}")
    try:
        interest_points[cid] = int(row['interest_points'])
    except Exception as e:
        raise ValueError(f"Invalid interest_points value for course_id {cid}: {row['interest_points']}")
m = gp.Model('OR_Course_Selection')
x_vars = m.addVars(course_ids, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((interest_points[cid] * x_vars[cid] for cid in course_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((credits[cid] * x_vars[cid] for cid in course_ids)) <= 20, name='credit_limit')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.0f} (Total interest points)')
    print('\n--- Selected Operations Research Courses ---')
    total_credits = 0
    for cid in course_ids:
        if x_vars[cid].X > 0.5:
            cname = ops_research_courses.loc[ops_research_courses['course_id'] == cid, 'course_name'].values[0]
            ccredits = credits[cid]
            cpoints = interest_points[cid]
            print(f'  {cid}: {cname} | Credits: {ccredits} | Interest Points: {cpoints}')
            total_credits += ccredits
    print(f'\nTotal credits used: {total_credits} / 20')
else:
    print(f'No optimal solution found. Status: {m.status}')