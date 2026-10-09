import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/50pct/S1/Large-scale-or/Mixture_testing/Mixture11/courses_42.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
discipline_col = 'discipline'
or_mask = df[discipline_col].str.strip().str.casefold() == 'operations research'
or_courses_df = df[or_mask].copy()
if or_courses_df.empty:
    raise ValueError("No courses found in the 'Operations Research' discipline.")
course_ids = or_courses_df['course_id'].tolist()
try:
    credits = {row['course_id']: int(row['credits']) for (_, row) in or_courses_df.iterrows()}
    interest_points = {row['course_id']: int(row['interest_points']) for (_, row) in or_courses_df.iterrows()}
except Exception as e:
    raise ValueError(f'Error converting credits or interest_points to int: {e}')
m = gp.Model('OR_Course_Selection')
x_vars = m.addVars(course_ids, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((interest_points[cid] * x_vars[cid] for cid in course_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((credits[cid] * x_vars[cid] for cid in course_ids)) <= 20, name='credit_limit')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    selected = [cid for cid in course_ids if x_vars[cid].X > 0.5]
    total_credits = sum((credits[cid] for cid in selected))
    total_interest = sum((interest_points[cid] for cid in selected))
    print(f'Optimal total interest points: {m.objVal}')
    print(f'Total credits used: {total_credits}')
    print('Selected Operations Research courses:')
    for cid in selected:
        cname = or_courses_df.loc[or_courses_df['course_id'] == cid, 'course_name'].values[0]
        print(f'  {cid}: {cname} (Credits: {credits[cid]}, Interest Points: {interest_points[cid]})')
else:
    print(f'No optimal solution found. Status: {m.status}')