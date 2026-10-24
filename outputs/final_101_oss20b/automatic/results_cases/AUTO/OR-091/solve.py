CSVQA_DATA = {'query': 'You are given a catalog of 42 university courses across six disciplines: Literature, Mathematics, Physics, Operations Research, Computer Science, and Chemistry. Each course has an associated number of credits and an “interest points” score. All the data is contained in the file courses_42.csv. The student wants to enroll in a subset of courses from the "Operations Research" discipline. The goal is to choose a combination of Operations Research courses whose total credits no more than 20 and whose total interest points are maximized.', 'relationships': [], 'tables': [{'columns': ['course_id', 'credits', 'interest_points', 'discipline'], 'file_name': 'courses_42.csv', 'records': [{'data_row_number': 21, 'values': {'course_id': 'C22', 'credits': '5', 'discipline': 'Operations Research', 'interest_points': '95'}}, {'data_row_number': 22, 'values': {'course_id': 'C23', 'credits': '5', 'discipline': 'Operations Research', 'interest_points': '92'}}, {'data_row_number': 23, 'values': {'course_id': 'C24', 'credits': '4', 'discipline': 'Operations Research', 'interest_points': '86'}}, {'data_row_number': 24, 'values': {'course_id': 'C25', 'credits': '4', 'discipline': 'Operations Research', 'interest_points': '82'}}, {'data_row_number': 25, 'values': {'course_id': 'C26', 'credits': '4', 'discipline': 'Operations Research', 'interest_points': '85'}}, {'data_row_number': 26, 'values': {'course_id': 'C27', 'credits': '4', 'discipline': 'Operations Research', 'interest_points': '80'}}, {'data_row_number': 27, 'values': {'course_id': 'C28', 'credits': '4', 'discipline': 'Operations Research', 'interest_points': '88'}}], 'table_id': 'file_0_view_0'}]}
import gurobipy
from gurobipy import GRB

def solve_problem():
    table = None
    for tbl in CSVQA_DATA['tables']:
        if tbl.get('table_id') == 'file_0_view_0':
            table = tbl
            break
    if table is None:
        raise ValueError("Required table 'file_0_view_0' not found in CSVQA_DATA")
    credits = {}
    interest = {}
    for rec in table['records']:
        vals = rec['values']
        cid = vals['course_id']
        credits[cid] = int(vals['credits'])
        interest[cid] = int(vals['interest_points'])
    for cid in credits:
        if cid not in interest:
            raise ValueError(f'Missing interest points for course {cid}')
    I = list(credits.keys())
    m = gurobipy.Model('OperationsResearchCourseSelection')
    m.setParam('MIPGap', 1e-06)
    x = m.addVars(I, vtype=GRB.BINARY, name='')
    m.setObjective(gurobipy.quicksum((interest[i] * x[i] for i in I)), GRB.MAXIMIZE)
    m.addConstr(gurobipy.quicksum((credits[i] * x[i] for i in I)) <= 20, name='CreditLimit')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'Optimal objective value: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName} = {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()
