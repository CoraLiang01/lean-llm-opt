import gurobipy as gp
import pandas as pd
import numpy as np
courses_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete 20260928/50pct/S3/Large-scale-or/Mixture_testing/Mixture11/courses_42.csv'
df = pd.read_csv(courses_path, sep=',')
discipline_col = 'discipline'
ops_research_mask = df[discipline_col].astype(str).str.casefold().str.strip() == 'operations research'
ops_df = df[ops_research_mask].copy()
if ops_df.empty:
    raise ValueError("No courses found in the 'Operations Research' discipline.")
C = list(ops_df['course_id'])
interest_points = dict(zip(ops_df['course_id'], ops_df['interest_points']))
credits = dict(zip(ops_df['course_id'], ops_df['credits']))
for cid in C:
    if pd.isnull(interest_points[cid]) or pd.isnull(credits[cid]):
        raise ValueError(f'Missing data for course_id {cid}.')
m = gp.Model('OR_Course_Selection')
x = m.addVars(C, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((interest_points[i] * x[i] for i in C)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((credits[i] * x[i] for i in C)) <= 20, name='credit_limit')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    selected_courses = [i for i in C if x[i].X > 0.5]
    total_interest = sum((interest_points[i] for i in selected_courses))
    total_credits = sum((credits[i] for i in selected_courses))
    print(f'Optimal total value/cost: {m.objVal} (Total interest points)')
    print(f'Total credits used: {total_credits} / 20')
    print(f'Selected Operations Research courses ({len(selected_courses)}):')
    for i in selected_courses:
        cname = ops_df.loc[ops_df['course_id'] == i, 'course_name'].values[0]
        print(f'  {i}: {cname} | Credits: {credits[i]}, Interest Points: {interest_points[i]}')
else:
    print(f'No optimal solution found. Status: {m.status}')