import gurobipy as gp
import pandas as pd
import numpy as np
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete 20260928/50pct/S1/Large-scale-or/Mixture_testing/Mixture11/courses_42.csv'
df = pd.read_csv(csv_path, sep=',')
or_mask = df['discipline'].astype(str).str.casefold().str.strip() == 'operations research'
df_or = df[or_mask].copy()
if df_or.empty:
    raise ValueError("No courses found in the 'Operations Research' discipline.")
courses_or = df_or['course_id'].astype(str).tolist()
credits = df_or.set_index('course_id')['credits'].astype(int).to_dict()
interest_points = df_or.set_index('course_id')['interest_points'].astype(int).to_dict()
for cid in courses_or:
    if cid not in credits or cid not in interest_points:
        raise ValueError(f'Missing credits or interest_points for course_id {cid}')
m = gp.Model('OR_Course_Selection')
x = m.addVars(courses_or, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((interest_points[cid] * x[cid] for cid in courses_or)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((credits[cid] * x[cid] for cid in courses_or)) <= 20, name='credit_limit')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    selected = [cid for cid in courses_or if x[cid].X > 0.5]
    total_credits = sum((credits[cid] for cid in selected))
    total_interest = sum((interest_points[cid] for cid in selected))
    print(f'Optimal total interest points: {m.objVal}')
    print(f'Total credits used: {total_credits}')
    print(f'Selected Operations Research courses (course_id, course_name, credits, interest_points):')
    for cid in selected:
        row = df_or[df_or['course_id'] == cid].iloc[0]
        print(f"  {cid}: {row['course_name']} | Credits: {row['credits']} | Interest Points: {row['interest_points']}")
else:
    print(f'No optimal solution found. Status: {m.status}')