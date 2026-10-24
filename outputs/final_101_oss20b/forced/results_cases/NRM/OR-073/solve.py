CSVQA_DATA = {'query': 'A factory produces three products, I, II, and III. Each product goes through two processing procedures, A and B. The factory has two types of equipment, A1 and A2, to complete procedure A, and three types of equipment, B1, B2, and B3, to complete procedure B. Product I can be processed on either type of A equipment or any type of B equipment. Product II can be processed on any type of A equipment, but when completing procedure B, it can only be processed on B1 equipment. Product III can only be processed on A2 and B2 equipment. Given the processing time, raw material cost, product selling price, available equipment operating time, and equipment cost at full load for each type of equipment, as shown in 43.csv, determine the optimal production plan to maximize profit. All production quantities should be continuous.', 'relationships': [], 'tables': [{'columns': ['Equipment / Cost', 'Product I', 'Product II', 'Product III', 'Available Equipment Operating Time', 'Equipment Cost at Full Load (yuan)'], 'file_name': '43.csv', 'records': [{'data_row_number': 0, 'values': {'Available Equipment Operating Time': '6000', 'Equipment / Cost': 'A1', 'Equipment Cost at Full Load (yuan)': '300', 'Product I': '5', 'Product II': '10', 'Product III': ''}}, {'data_row_number': 1, 'values': {'Available Equipment Operating Time': '10000', 'Equipment / Cost': 'A2', 'Equipment Cost at Full Load (yuan)': '321', 'Product I': '7', 'Product II': '9', 'Product III': '12'}}, {'data_row_number': 2, 'values': {'Available Equipment Operating Time': '8000', 'Equipment / Cost': 'A3', 'Equipment Cost at Full Load (yuan)': '203', 'Product I': '6', 'Product II': '11', 'Product III': '2'}}, {'data_row_number': 3, 'values': {'Available Equipment Operating Time': '4000', 'Equipment / Cost': 'B1', 'Equipment Cost at Full Load (yuan)': '250', 'Product I': '6', 'Product II': '8', 'Product III': ''}}, {'data_row_number': 4, 'values': {'Available Equipment Operating Time': '7000', 'Equipment / Cost': 'B2', 'Equipment Cost at Full Load (yuan)': '783', 'Product I': '4', 'Product II': '', 'Product III': '11'}}, {'data_row_number': 5, 'values': {'Available Equipment Operating Time': '4000', 'Equipment / Cost': 'B3', 'Equipment Cost at Full Load (yuan)': '200', 'Product I': '7', 'Product II': '', 'Product III': ''}}, {'data_row_number': 6, 'values': {'Available Equipment Operating Time': '5000', 'Equipment / Cost': 'B4', 'Equipment Cost at Full Load (yuan)': '300', 'Product I': '3', 'Product II': '5', 'Product III': '8'}}, {'data_row_number': 7, 'values': {'Available Equipment Operating Time': '', 'Equipment / Cost': 'Raw Material Cost (yuan/unit)', 'Equipment Cost at Full Load (yuan)': '', 'Product I': '0.25', 'Product II': '0.35', 'Product III': '0.5'}}, {'data_row_number': 8, 'values': {'Available Equipment Operating Time': '', 'Equipment / Cost': 'Unit Price (yuan/unit)', 'Equipment Cost at Full Load (yuan)': '', 'Product I': '1.25', 'Product II': '2', 'Product III': '2.8'}}], 'table_id': 'file_0_view_0'}]}
import gurobipy
from gurobipy import GRB

def solve_problem():
    P = ['I', 'II', 'III']
    A = ['A1', 'A2', 'A3']
    B = ['B1', 'B2', 'B3', 'B4']
    E = A + B
    table = None
    for tbl in CSVQA_DATA['tables']:
        if tbl.get('table_id') == 'file_0_view_0':
            table = tbl
            break
    if table is None:
        raise ValueError("Required table 'file_0_view_0' not found in CSVQA_DATA")
    T_e = {}
    c_e = {}
    t = {p: {} for p in P}
    r_p = {}
    p_p = {}

    def to_float(val):
        if isinstance(val, str):
            val = val.strip()
            if val == '':
                return None
        try:
            return float(val)
        except Exception:
            return None
    for rec in table['records']:
        values = rec['values']
        equip = values.get('Equipment / Cost', '').strip()
        if equip in ('Raw Material Cost (yuan/unit)', 'Unit Price (yuan/unit)'):
            if equip == 'Raw Material Cost (yuan/unit)':
                for p_col, p_key in zip(['Product I', 'Product II', 'Product III'], P):
                    val = to_float(values.get(p_col, ''))
                    if val is None:
                        raise ValueError(f'Missing raw material cost for product {p_key}')
                    r_p[p_key] = val
            else:
                for p_col, p_key in zip(['Product I', 'Product II', 'Product III'], P):
                    val = to_float(values.get(p_col, ''))
                    if val is None:
                        raise ValueError(f'Missing unit price for product {p_key}')
                    p_p[p_key] = val
            continue
        if equip == '':
            continue
        avail_time = to_float(values.get('Available Equipment Operating Time', ''))
        if avail_time is None:
            raise ValueError(f'Missing available time for equipment {equip}')
        T_e[equip] = avail_time
        cost_hour = to_float(values.get('Equipment Cost at Full Load (yuan)', ''))
        if cost_hour is None:
            raise ValueError(f'Missing equipment cost for equipment {equip}')
        c_e[equip] = cost_hour
        for p_col, p_key in zip(['Product I', 'Product II', 'Product III'], P):
            val = to_float(values.get(p_col, ''))
            if val is not None:
                t[p_key][equip] = val
    for e in E:
        if e not in T_e or e not in c_e:
            raise ValueError(f'Missing parameters for equipment {e}')
    for p in P:
        if p not in r_p or p not in p_p:
            raise ValueError(f'Missing cost or price for product {p}')
    allow = {}
    for p in P:
        for e in E:
            if p == 'I':
                allow[p, e] = True
            elif p == 'II':
                allow[p, e] = e in A or e == 'B1'
            elif p == 'III':
                allow[p, e] = e == 'A2' or e == 'B2'
            else:
                allow[p, e] = False
    m = gurobipy.Model()
    m.setParam('MIPGap', 1e-06)
    y = {}
    for (p, e), is_allowed in allow.items():
        if is_allowed:
            if e not in t[p]:
                raise ValueError(f'Processing time missing for product {p} on equipment {e}')
            y[p, e] = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f'y_{p}_{e}')
    m.update()
    obj_expr = gurobipy.quicksum((y[p, e] * (p_p[p] - r_p[p] - c_e[e] * t[p][e]) for p, e in y))
    m.setObjective(obj_expr, GRB.MAXIMIZE)
    for e in E:
        expr = gurobipy.quicksum((y[p, e] * t[p][e] for p in P if (p, e) in y))
        m.addConstr(expr <= T_e[e], name=f'time_limit_{e}')
    for p in P:
        sum_A = gurobipy.quicksum((y[p, e] for e in A if (p, e) in y))
        sum_B = gurobipy.quicksum((y[p, e] for e in B if (p, e) in y))
        m.addConstr(sum_A == sum_B, name=f'proc_consistency_{p}')
    m.optimize()
    return m
m = solve_problem()
