import gurobipy as gp
import pandas as pd
import numpy as np
courses_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture11/courses_42.csv', sep=',')
or_courses_df = courses_df[courses_df['discipline'].astype(str).str.casefold().str.strip() == 'operations research']
or_course_ids = or_courses_df['course_id'].astype(str).tolist()
credits = or_courses_df.set_index('course_id')['credits'].astype(int).to_dict()
interest_points = or_courses_df.set_index('course_id')['interest_points'].astype(int).to_dict()
for cid in or_course_ids:
    if cid not in credits or cid not in interest_points:
        raise ValueError(f'Missing data for Operations Research course_id: {cid}')
m = gp.Model('OR_Course_Selection')
x = m.addVars(or_course_ids, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((interest_points[cid] * x[cid] for cid in or_course_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((credits[cid] * x[cid] for cid in or_course_ids)) <= 20, name='credit_limit')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    selected = [cid for cid in or_course_ids if x[cid].X > 0.5]
    total_credits = sum((credits[cid] for cid in selected))
    total_interest = sum((interest_points[cid] for cid in selected))
    print(f'Optimal total interest points: {total_interest}')
    print(f'Total credits used: {total_credits}')
    print('Selected Operations Research courses:')
    for cid in selected:
        row = or_courses_df[or_courses_df['course_id'] == cid].iloc[0]
        print(f"  {cid}: {row['course_name']} (Credits: {row['credits']}, Interest Points: {row['interest_points']})")
else:
    print(f'No optimal solution found. Status: {m.status}')