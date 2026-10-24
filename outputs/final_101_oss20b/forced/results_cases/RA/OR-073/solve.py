CSVQA_DATA = {'query': 'A factory produces three products, I, II, and III. Each product goes through two processing procedures, A and B. The factory has two types of equipment, A1 and A2, to complete procedure A, and three types of equipment, B1, B2, and B3, to complete procedure B. Product I can be processed on either type of A equipment or any type of B equipment. Product II can be processed on any type of A equipment, but when completing procedure B, it can only be processed on B1 equipment. Product III can only be processed on A2 and B2 equipment. Given the processing time, raw material cost, product selling price, available equipment operating time, and equipment cost at full load for each type of equipment, as shown in 43.csv, determine the optimal production plan to maximize profit. All production quantities should be continuous.', 'relationships': [], 'tables': [{'columns': ['Equipment / Cost', 'Product I', 'Product II', 'Product III', 'Available Equipment Operating Time', 'Equipment Cost at Full Load (yuan)'], 'file_name': '43.csv', 'records': [{'data_row_number': 0, 'values': {'Available Equipment Operating Time': '6000', 'Equipment / Cost': 'A1', 'Equipment Cost at Full Load (yuan)': '300', 'Product I': '5', 'Product II': '10', 'Product III': ''}}, {'data_row_number': 1, 'values': {'Available Equipment Operating Time': '10000', 'Equipment / Cost': 'A2', 'Equipment Cost at Full Load (yuan)': '321', 'Product I': '7', 'Product II': '9', 'Product III': '12'}}, {'data_row_number': 3, 'values': {'Available Equipment Operating Time': '4000', 'Equipment / Cost': 'B1', 'Equipment Cost at Full Load (yuan)': '250', 'Product I': '6', 'Product II': '8', 'Product III': ''}}], 'table_id': 'file_0_view_0'}]}
import gurobipy
from gurobipy import GRB

def solve_problem():
    P = ('I', 'II', 'III')
    E = ('A1', 'A2', 'B1')
    allowed_pairs = {('I', 'A1'), ('I', 'A2'), ('I', 'B1'), ('II', 'A1'), ('II', 'A2'), ('II', 'B1'), ('III', 'A2')}
    pi = {'I': 0, 'II': 0, 'III': 0}
    for p in P:
        if p not in pi:
            raise ValueError(f'Missing profit parameter for product {p}')
    table = None
    for tbl in CSVQA_DATA.get('tables', []):
        if tbl.get('table_id') == 'file_0_view_0':
            table = tbl
            break
    if table is None:
        raise ValueError('Required table file_0_view_0 not found')
    A = {}
    c = {}
    t = {}
    for rec in table.get('records', []):
        vals = rec.get('values', {})
        equip = vals.get('Equipment / Cost')
        if equip not in E:
            continue
        try:
            A[equip] = float(vals.get('Available Equipment Operating Time', 0))
            c[equip] = float(vals.get('Equipment Cost at Full Load (yuan)', 0))
        except ValueError as e:
            raise ValueError(f'Invalid numeric value for equipment {equip}') from e
        for p in P:
            val_str = vals.get(p, '')
            if val_str == '':
                continue
            try:
                t[p, equip] = float(val_str)
            except ValueError as e:
                raise ValueError(f'Invalid processing time for product {p} on {equip}') from e
    for e in E:
        if e not in A:
            raise ValueError(f'Missing available time for equipment {e}')
        if e not in c:
            raise ValueError(f'Missing cost for equipment {e}')
    for pair in allowed_pairs:
        if pair not in t:
            raise ValueError(f'Missing processing time for pair {pair}')
    m = gurobipy.Model()
    m.setParam('MIPGap', 1e-06)
    x = m.addVars([(p, e) for p, e in allowed_pairs], lb=0, vtype=GRB.CONTINUOUS, name='')
    obj = gurobipy.quicksum((pi[p] * x[p, e] for p in P for e in E if (p, e) in allowed_pairs))
    obj -= gurobipy.quicksum((c[e] * t[p, e] * x[p, e] for p, e in allowed_pairs))
    m.setObjective(obj, GRB.MAXIMIZE)
    for e in E:
        expr = gurobipy.quicksum((t[p, e] * x[p, e] for p in P if (p, e) in allowed_pairs))
        m.addConstr(expr <= A[e], name=f'cap_{e}')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()
