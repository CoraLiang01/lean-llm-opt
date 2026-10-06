import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete 20260928/100pct/S1/Large-scale-or/Mixture_testing/Mixture11/courses_42.csv'
df = pd.read_csv(csv_path, sep=',')
discipline_col = 'discipline'
ops_research_mask = df[discipline_col].astype(str).str.casefold().str.strip() == 'operations research'
df_or = df[ops_research_mask].copy()
if df_or.empty:
    raise ValueError("No courses found in the 'Operations Research' discipline.")
course_ids = df_or['course_id'].astype(str).tolist()
credits = df_or.set_index('course_id')['credits'].astype(int).to_dict()
interest_points = df_or.set_index('course_id')['interest_points'].astype(int).to_dict()
for cid in course_ids:
    if cid not in credits or cid not in interest_points:
        raise ValueError(f'Missing credits or interest_points for course_id {cid}')
m = gp.Model('OR_Course_Selection')
x = m.addVars(course_ids, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((interest_points[cid] * x[cid] for cid in course_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((credits[cid] * x[cid] for cid in course_ids)) <= 20, name='credit_limit')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    selected = [cid for cid in course_ids if x[cid].X > 0.5]
    total_credits = sum((credits[cid] for cid in selected))
    total_interest = sum((interest_points[cid] for cid in selected))
    print(f'Optimal total interest points: {m.objVal:.0f}')
    print(f'Total credits used: {total_credits} / 20')
    print(f'Selected Operations Research courses ({len(selected)}):')
    for cid in selected:
        cname = df_or.loc[df_or['course_id'] == cid, 'course_name'].values[0]
        print(f'  {cid}: {cname} (Credits: {credits[cid]}, Interest: {interest_points[cid]})')
else:
    print(f'No optimal solution found. Status: {m.status}')