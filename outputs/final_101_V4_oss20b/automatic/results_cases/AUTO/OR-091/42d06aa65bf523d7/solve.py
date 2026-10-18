import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
import json
courses_path = '/root/rivermind-data/lean-gptoss20b-unified-rag-20260928/source/Test_Dataset/Large-scale-or/Mixture_testing/Mixture11/courses_42.csv'
df = pd.read_csv(courses_path, sep=',')
or_df = df[df['discipline'].str.lower() == 'operations research']
OR = list(or_df['course_id'])
credits = dict(zip(or_df['course_id'], or_df['credits']))
interest_points = dict(zip(or_df['course_id'], or_df['interest_points']))
credit_limit = 20
m = gp.Model('OR_Course_Selection')
x = m.addVars(OR, vtype=gp.GRB.BINARY)
m.setObjective(gp.quicksum((interest_points[i] * x[i] for i in OR)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((credits[i] * x[i] for i in OR)) <= credit_limit)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    total_interest = m.objVal
    total_credits = sum((credits[i] * x[i].X for i in OR))
    print(f'Optimal total interest points: {total_interest:.0f}')
    print(f'Total credits used: {total_credits:.0f} / {credit_limit}')
    print('\nSelected courses:')
    for i in OR:
        if x[i].X > 0.5:
            print(f'  {i}: {or_df[or_df['course_id'] == i]['course_name'].values[0]} (Credits: {credits[i]}, Interest: {interest_points[i]})')
else:
    print(f'No optimal solution found. Status: {m.status}')