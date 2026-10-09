import gurobipy as gp
import pandas as pd
import numpy as np
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture11/courses_42.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
discipline_norm = df['discipline'].str.strip().str.casefold()
ops_research_mask = discipline_norm == 'operations research'
ops_df = df[ops_research_mask].copy()
if ops_df.empty:
    raise ValueError("No courses found in the 'Operations Research' discipline.")
ops_df['credits'] = ops_df['credits'].astype(int)
ops_df['interest_points'] = ops_df['interest_points'].astype(int)
course_ids = ops_df['course_id'].tolist()
credits = dict(zip(ops_df['course_id'], ops_df['credits']))
interest_points = dict(zip(ops_df['course_id'], ops_df['interest_points']))
m = gp.Model('OR_Course_Selection')
x_vars = m.addVars(course_ids, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((interest_points[cid] * x_vars[cid] for cid in course_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((credits[cid] * x_vars[cid] for cid in course_ids)) <= 20, name='credit_limit')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    selected = [cid for cid in course_ids if x_vars[cid].X > 0.5]
    total_credits = sum((credits[cid] for cid in selected))
    total_interest = sum((interest_points[cid] for cid in selected))
    print(f'Optimal total interest points: {total_interest}')
    print(f'Total credits used: {total_credits}')
    print('Selected Operations Research courses:')
    for cid in selected:
        cname = ops_df.loc[ops_df['course_id'] == cid, 'course_name'].values[0]
        print(f'  {cid}: {cname} (Credits: {credits[cid]}, Interest Points: {interest_points[cid]})')
else:
    print(f'No optimal solution found. Status: {m.status}')