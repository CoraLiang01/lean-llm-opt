import gurobipy as gp
import pandas as pd
import numpy as np
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/100pct/S1/Large-scale-or/Mixture_testing/Mixture11/courses_42.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
discipline_col = 'discipline'
or_mask = df[discipline_col].str.strip().str.casefold() == 'operations research'
or_courses_df = df[or_mask].copy()
if or_courses_df.empty:
    raise ValueError("No courses found in the 'Operations Research' discipline.")
course_id_col = 'course_id'
credits_col = 'credits'
interest_col = 'interest_points'
C = list(or_courses_df[course_id_col])
credits = {}
interest_points = {}
for (idx, row) in or_courses_df.iterrows():
    cid = row[course_id_col]
    try:
        credits[cid] = int(row[credits_col])
    except Exception:
        raise ValueError(f'Invalid credits value for course_id {cid}: {row[credits_col]}')
    try:
        interest_points[cid] = int(row[interest_col])
    except Exception:
        raise ValueError(f'Invalid interest_points value for course_id {cid}: {row[interest_col]}')
m = gp.Model('OR_Course_Selection')
x_vars = m.addVars(C, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((interest_points[c] * x_vars[c] for c in C)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((credits[c] * x_vars[c] for c in C)) <= 20, name='credit_limit')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    selected_courses = [c for c in C if x_vars[c].X > 0.5]
    total_credits = sum((credits[c] for c in selected_courses))
    total_interest = sum((interest_points[c] for c in selected_courses))
    print(f'Optimal total interest points: {m.objVal}')
    print(f'Total credits used: {total_credits}')
    print('Selected Operations Research courses:')
    for c in selected_courses:
        cname = or_courses_df.loc[or_courses_df[course_id_col] == c, 'course_name'].values[0]
        print(f'  {c}: {cname} (Credits: {credits[c]}, Interest Points: {interest_points[c]})')
else:
    print(f'No optimal solution found. Status: {m.status}')