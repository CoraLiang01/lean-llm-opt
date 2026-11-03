import gurobipy as gp
import pandas as pd
import numpy as np
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture11/courses_42.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
discipline_col = 'discipline'
ops_research_mask = df[discipline_col].str.strip().str.casefold() == 'operations research'
ops_research_df = df[ops_research_mask].copy()
if ops_research_df.empty:
    raise ValueError("No courses found in the 'Operations Research' discipline.")
ops_course_ids = ops_research_df['course_id'].tolist()
credits = {}
interest_points = {}
for idx, row in ops_research_df.iterrows():
    cid = row['course_id']
    try:
        credits[cid] = int(row['credits'])
        interest_points[cid] = int(row['interest_points'])
    except Exception as e:
        raise ValueError(f'Invalid numeric value for course_id {cid}: {e}')
m = gp.Model('OR_Course_Selection')
x_vars = m.addVars(ops_course_ids, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((interest_points[cid] * x_vars[cid] for cid in ops_course_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((credits[cid] * x_vars[cid] for cid in ops_course_ids)) <= 20, name='credit_limit')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total interest points: {m.objVal:.0f}')
    print('Selected Operations Research courses:')
    total_credits = 0
    for cid in ops_course_ids:
        if x_vars[cid].X > 0.5:
            cname = ops_research_df.loc[ops_research_df['course_id'] == cid, 'course_name'].values[0]
            ccredits = credits[cid]
            cpoints = interest_points[cid]
            print(f'  {cid}: {cname} | Credits: {ccredits} | Interest Points: {cpoints}')
            total_credits += ccredits
    print(f'Total credits used: {total_credits}')
else:
    print(f'No optimal solution found. Status: {m.status}')