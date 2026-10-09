import gurobipy as gp
import pandas as pd
import numpy as np
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/100pct/S1/Large-scale-or/Mixture_testing/Mixture11/courses_42.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
discipline_norm = df['discipline'].str.strip().str.casefold()
ops_research_mask = discipline_norm == 'operations research'
ops_research_df = df[ops_research_mask].copy()
if ops_research_df.empty:
    raise ValueError("No courses found in the 'Operations Research' discipline.")
course_ids = ops_research_df['course_id'].tolist()
try:
    credits = ops_research_df.set_index('course_id')['credits'].astype(int).to_dict()
    interest_points = ops_research_df.set_index('course_id')['interest_points'].astype(int).to_dict()
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
        cname = ops_research_df.loc[ops_research_df['course_id'] == cid, 'course_name'].values[0]
        print(f'  {cid}: {cname} (Credits: {credits[cid]}, Interest: {interest_points[cid]})')
else:
    print(f'No optimal solution found. Status: {m.status}')