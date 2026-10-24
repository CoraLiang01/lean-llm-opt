CSVQA_DATA = {'query': 'You are given a catalog of 42 university courses across six disciplines: Literature, Mathematics, Physics, Operations Research, Computer Science, and Chemistry. Each course has an associated number of credits and an “interest points” score. All the data is contained in the file courses_42.csv. The student wants to enroll in a subset of courses from the "Operations Research" discipline. The goal is to choose a combination of Operations Research courses whose total credits no more than 20 and whose total interest points are maximized.', 'relationships': [], 'tables': [{'columns': ['course_id', 'credits', 'interest_points', 'discipline'], 'file_name': 'courses_42.csv', 'records': [{'data_row_number': 21, 'values': {'course_id': 'C22', 'credits': '5', 'discipline': 'Operations Research', 'interest_points': '95'}}, {'data_row_number': 22, 'values': {'course_id': 'C23', 'credits': '5', 'discipline': 'Operations Research', 'interest_points': '92'}}, {'data_row_number': 23, 'values': {'course_id': 'C24', 'credits': '4', 'discipline': 'Operations Research', 'interest_points': '86'}}, {'data_row_number': 24, 'values': {'course_id': 'C25', 'credits': '4', 'discipline': 'Operations Research', 'interest_points': '82'}}, {'data_row_number': 25, 'values': {'course_id': 'C26', 'credits': '4', 'discipline': 'Operations Research', 'interest_points': '85'}}, {'data_row_number': 26, 'values': {'course_id': 'C27', 'credits': '4', 'discipline': 'Operations Research', 'interest_points': '80'}}, {'data_row_number': 27, 'values': {'course_id': 'C28', 'credits': '4', 'discipline': 'Operations Research', 'interest_points': '88'}}], 'table_id': 'file_0_view_0'}]}
import gurobipy
from gurobipy import GRB

def solve_problem():
    tables = CSVQA_DATA['tables']
    target_table = None
    for tbl in tables:
        if tbl.get('table_id') == 'file_0_view_0':
            target_table = tbl
            break
    if target_table is None:
        raise ValueError("Required table 'file_0_view_0' not found in CSVQA_DATA")
    records = target_table.get('records', [])
    if not records:
        raise ValueError('No records found in the target table')
    ids = []
    credits = {}
    interest = {}
    for rec in records:
        vals = rec.get('values', {})
        course_id = vals.get('course_id')
        if course_id is None:
            raise ValueError('Missing course_id in record')
        cred_str = vals.get('credits')
        ip_str = vals.get('interest_points')
        if cred_str is None or ip_str is None:
            raise ValueError(f'Missing credits or interest_points for course {course_id}')
        try:
            cred = int(cred_str)
            ip = int(ip_str)
        except ValueError:
            raise ValueError(f'Non-integer credits or interest_points for course {course_id}')
        ids.append(course_id)
        credits[course_id] = cred
        interest[course_id] = ip
    m = gurobipy.Model('OperationsResearchCourseSelection')
    m.setParam('MIPGap', 1e-06)
    x = m.addVars(ids, vtype=GRB.BINARY, name='')
    m.setObjective(gurobipy.quicksum((interest[i] * x[i] for i in ids)), GRB.MAXIMIZE)
    m.addConstr(gurobipy.quicksum((credits[i] * x[i] for i in ids)) <= 20, name='credit_limit')
    m.optimize()
    return m
m = solve_problem()
if m.Status == GRB.OPTIMAL:
    print(m.ObjVal)
    for var in m.getVars():
        print(var.VarName, var.X)
else:
    print(m.Status)
