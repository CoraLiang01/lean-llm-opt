CSVQA_DATA = {'query': 'You are given a catalog of 42 university courses across six disciplines: Literature, Mathematics, Physics, Operations Research, Computer Science, and Chemistry. Each course has an associated number of credits and an “interest points” score. All the data is contained in the file courses_42.csv. The student wants to enroll in a subset of courses from the "Operations Research" discipline. The goal is to choose a combination of Operations Research courses whose total credits no more than 20 and whose total interest points are maximized.', 'relationships': [], 'tables': [{'columns': ['course_id', 'credits', 'interest_points', 'discipline'], 'file_name': 'courses_42.csv', 'records': [{'data_row_number': 21, 'values': {'course_id': 'C22', 'credits': '5', 'discipline': 'Operations Research', 'interest_points': '95'}}, {'data_row_number': 22, 'values': {'course_id': 'C23', 'credits': '5', 'discipline': 'Operations Research', 'interest_points': '92'}}, {'data_row_number': 23, 'values': {'course_id': 'C24', 'credits': '4', 'discipline': 'Operations Research', 'interest_points': '86'}}, {'data_row_number': 24, 'values': {'course_id': 'C25', 'credits': '4', 'discipline': 'Operations Research', 'interest_points': '82'}}, {'data_row_number': 25, 'values': {'course_id': 'C26', 'credits': '4', 'discipline': 'Operations Research', 'interest_points': '85'}}, {'data_row_number': 26, 'values': {'course_id': 'C27', 'credits': '4', 'discipline': 'Operations Research', 'interest_points': '80'}}, {'data_row_number': 27, 'values': {'course_id': 'C28', 'credits': '4', 'discipline': 'Operations Research', 'interest_points': '88'}}], 'table_id': 'file_0_view_0'}]}
import gurobipy
from gurobipy import GRB

def solve_problem():
    tables = CSVQA_DATA.get('tables', [])
    table = next((t for t in tables if t.get('table_id') == 'file_0_view_0'), None)
    if table is None:
        raise ValueError("Operations Research table 'file_0_view_0' not found in CSVQA_DATA")
    course_ids = []
    credits = {}
    interest_points = {}
    for record in table.get('records', []):
        values = record.get('values', {})
        for field in ('course_id', 'credits', 'interest_points'):
            if field not in values:
                raise ValueError(f"Missing field '{field}' in record {record.get('data_row_number')}")
        cid = values['course_id']
        try:
            cred = int(values['credits'])
            ip = int(values['interest_points'])
        except ValueError as e:
            raise ValueError(f"Non-numeric value in record {record.get('data_row_number')}: {e}")
        if cid in course_ids:
            raise ValueError(f"Duplicate course_id '{cid}' found")
        course_ids.append(cid)
        credits[cid] = cred
        interest_points[cid] = ip
    m = gurobipy.Model()
    m.setParam('MIPGap', 1e-06)
    y = m.addVars(course_ids, vtype=GRB.BINARY, name='')
    m.setObjective(gurobipy.quicksum((interest_points[i] * y[i] for i in course_ids)), GRB.MAXIMIZE)
    m.addConstr(gurobipy.quicksum((credits[i] * y[i] for i in course_ids)) <= 20, name='credit_limit')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()
