import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/100pct/S2/Large-scale-or/Mixture_testing/Mixture11/courses_42.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
discipline_col = 'discipline'
ops_research_mask = df[discipline_col].str.strip().str.casefold() == 'operations research'
ops_research_df = df[ops_research_mask].copy()
if ops_research_df.empty:
    raise ValueError("No courses found in the 'Operations Research' discipline.")
course_id_col = 'course_id'
interest_points_col = 'interest_points'
credits_col = 'credits'
ops_course_ids = ops_research_df[course_id_col].tolist()
try:
    interest_points = ops_research_df.set_index(course_id_col)[interest_points_col].astype(int).to_dict()
except Exception as e:
    raise ValueError(f"Error converting 'interest_points' to int: {e}")
try:
    credits = ops_research_df.set_index(course_id_col)[credits_col].astype(int).to_dict()
except Exception as e:
    raise ValueError(f"Error converting 'credits' to int: {e}")
for cid in ops_course_ids:
    if cid not in interest_points or cid not in credits:
        raise ValueError(f'Missing interest_points or credits for course_id {cid}')
m = gp.Model('OR_Course_Selection')
x_vars = m.addVars(ops_course_ids, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((interest_points[cid] * x_vars[cid] for cid in ops_course_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((credits[cid] * x_vars[cid] for cid in ops_course_ids)) <= 20, name='credit_limit')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    selected_courses = [cid for cid in ops_course_ids if x_vars[cid].X > 0.5]
    total_interest = sum((interest_points[cid] for cid in selected_courses))
    total_credits = sum((credits[cid] for cid in selected_courses))
    print(f'Optimal total value/cost: {m.objVal:.0f} (Total Interest Points)')
    print(f'Total credits used: {total_credits} / 20')
    print('\n--- Selected Operations Research Courses ---')
    for cid in selected_courses:
        row = ops_research_df[ops_research_df[course_id_col] == cid].iloc[0]
        print(f"  {cid}: {row['course_name']} | Credits: {credits[cid]} | Interest Points: {interest_points[cid]}")
else:
    print(f'No optimal solution found. Status: {m.status}')