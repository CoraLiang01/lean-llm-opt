CSVQA_DATA = {'query': 'You are given a catalog of 42 university courses across six disciplines: Literature, Mathematics, Physics, Operations Research, Computer Science, and Chemistry. Each course has an associated number of credits and an “interest points” score. All the data is contained in the file courses_42.csv. The student wants to enroll in a subset of courses from the "Operations Research" discipline. The goal is to choose a combination of Operations Research courses whose total credits no more than 20 and whose total interest points are maximized.', 'relationships': [], 'tables': [{'columns': ['course_id', 'credits', 'interest_points', 'discipline'], 'file_name': 'courses_42.csv', 'records': [{'data_row_number': 21, 'values': {'course_id': 'C22', 'credits': '5', 'discipline': 'Operations Research', 'interest_points': '95'}}, {'data_row_number': 22, 'values': {'course_id': 'C23', 'credits': '5', 'discipline': 'Operations Research', 'interest_points': '92'}}, {'data_row_number': 23, 'values': {'course_id': 'C24', 'credits': '4', 'discipline': 'Operations Research', 'interest_points': '86'}}, {'data_row_number': 24, 'values': {'course_id': 'C25', 'credits': '4', 'discipline': 'Operations Research', 'interest_points': '82'}}, {'data_row_number': 25, 'values': {'course_id': 'C26', 'credits': '4', 'discipline': 'Operations Research', 'interest_points': '85'}}, {'data_row_number': 26, 'values': {'course_id': 'C27', 'credits': '4', 'discipline': 'Operations Research', 'interest_points': '80'}}, {'data_row_number': 27, 'values': {'course_id': 'C28', 'credits': '4', 'discipline': 'Operations Research', 'interest_points': '88'}}], 'table_id': 'file_0_view_0'}]}
import gurobipy
from gurobipy import GRB

def solve_problem():
    tables = CSVQA_DATA.get('tables', [])
    view_table = None
    for tbl in tables:
        if tbl.get('table_id') == 'file_0_view_0':
            view_table = tbl
            break
    if view_table is None:
        raise ValueError("Required table 'file_0_view_0' not found in CSVQA_DATA")
    records = view_table.get('records', [])
    if not records:
        raise ValueError('No records found in the Operations Research view')
    course_ids = []
    credits_dict = {}
    interest_dict = {}
    for rec in records:
        vals = rec.get('values', {})
        cid = vals.get('course_id')
        if cid is None:
            raise ValueError(f"Missing course_id in record {rec.get('data_row_number')}")
        credits_str = vals.get('credits')
        interest_str = vals.get('interest_points')
        if credits_str is None or interest_str is None:
            raise ValueError(f'Missing credits or interest_points for course {cid}')
        try:
            credits = int(credits_str)
            interest = int(interest_str)
        except ValueError as e:
            raise ValueError(f'Non-integer parameter for course {cid}: {e}')
        course_ids.append(cid)
        credits_dict[cid] = credits
        interest_dict[cid] = interest
    if set(course_ids) != set(credits_dict.keys()):
        missing = set(course_ids) - set(credits_dict.keys())
        raise ValueError(f'Missing credits for courses: {missing}')
    if set(course_ids) != set(interest_dict.keys()):
        missing = set(course_ids) - set(interest_dict.keys())
        raise ValueError(f'Missing interest_points for courses: {missing}')
    model = gurobipy.Model()
    model.setParam('MIPGap', 1e-06)
    x = model.addVars(course_ids, vtype=GRB.BINARY, name='')
    model.setObjective(gurobipy.quicksum((interest_dict[i] * x[i] for i in course_ids)), GRB.MAXIMIZE)
    model.addConstr(gurobipy.quicksum((credits_dict[i] * x[i] for i in course_ids)) <= 20, name='credit_limit')
    model.optimize()
    return model
m = solve_problem()
