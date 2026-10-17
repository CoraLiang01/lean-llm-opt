CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'A factory produces three products, I, II, and III. Each product goes through two processing procedures, A '
          'and B. The factory has two types of equipment, A1 and A2, to complete procedure A, and three types of '
          'equipment, B1, B2, and B3, to complete procedure B. Product I can be processed on either type of A '
          'equipment or any type of B equipment. Product II can be processed on any type of A equipment, but when '
          'completing procedure B, it can only be processed on B1 equipment. Product III can only be processed on A2 '
          'and B2 equipment. Given the processing time, raw material cost, product selling price, available equipment '
          'operating time, and equipment cost at full load for each type of equipment, as shown in 43.csv, determine '
          'the optimal production plan to maximize profit. All production quantities should be continuous.',
 'relationships': [],
 'route': 'Others',
 'tables': [{'columns': ['Equipment / Cost',
                         'Product I',
                         'Product II',
                         'Product III',
                         'Available Equipment Operating Time',
                         'Equipment Cost at Full Load (yuan)'],
             'file_index': 0,
             'file_name': '43.csv',
             'filters': {},
             'original_rows': 9,
             'records': [{'source_row': 0,
                          'values': {'Available Equipment Operating Time': '6000',
                                     'Equipment / Cost': 'A1',
                                     'Equipment Cost at Full Load (yuan)': '300',
                                     'Product I': '5',
                                     'Product II': '10',
                                     'Product III': ''}},
                         {'source_row': 1,
                          'values': {'Available Equipment Operating Time': '10000',
                                     'Equipment / Cost': 'A2',
                                     'Equipment Cost at Full Load (yuan)': '321',
                                     'Product I': '7',
                                     'Product II': '9',
                                     'Product III': '12'}},
                         {'source_row': 2,
                          'values': {'Available Equipment Operating Time': '8000',
                                     'Equipment / Cost': 'A3',
                                     'Equipment Cost at Full Load (yuan)': '203',
                                     'Product I': '6',
                                     'Product II': '11',
                                     'Product III': '2'}},
                         {'source_row': 3,
                          'values': {'Available Equipment Operating Time': '4000',
                                     'Equipment / Cost': 'B1',
                                     'Equipment Cost at Full Load (yuan)': '250',
                                     'Product I': '6',
                                     'Product II': '8',
                                     'Product III': ''}},
                         {'source_row': 4,
                          'values': {'Available Equipment Operating Time': '7000',
                                     'Equipment / Cost': 'B2',
                                     'Equipment Cost at Full Load (yuan)': '783',
                                     'Product I': '4',
                                     'Product II': '',
                                     'Product III': '11'}},
                         {'source_row': 5,
                          'values': {'Available Equipment Operating Time': '4000',
                                     'Equipment / Cost': 'B3',
                                     'Equipment Cost at Full Load (yuan)': '200',
                                     'Product I': '7',
                                     'Product II': '',
                                     'Product III': ''}},
                         {'source_row': 6,
                          'values': {'Available Equipment Operating Time': '5000',
                                     'Equipment / Cost': 'B4',
                                     'Equipment Cost at Full Load (yuan)': '300',
                                     'Product I': '3',
                                     'Product II': '5',
                                     'Product III': '8'}},
                         {'source_row': 7,
                          'values': {'Available Equipment Operating Time': '',
                                     'Equipment / Cost': 'Raw Material Cost (yuan/unit)',
                                     'Equipment Cost at Full Load (yuan)': '',
                                     'Product I': '0.25',
                                     'Product II': '0.35',
                                     'Product III': '0.5'}},
                         {'source_row': 8,
                          'values': {'Available Equipment Operating Time': '',
                                     'Equipment / Cost': 'Unit Price (yuan/unit)',
                                     'Equipment Cost at Full Load (yuan)': '',
                                     'Product I': '1.25',
                                     'Product II': '2',
                                     'Product III': '2.8'}}],
             'returned_rows': 9,
             'role': 'descriptive non-unique role',
             'table_id': 'file_0_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
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
    P = ['I', 'II', 'III']
    E_A = ['A1', 'A2']
    E_B = ['B1', 'B2', 'B3']
    tA = {(p, e): None for p in P for e in E_A}
    tB = {(p, e): None for p in P for e in E_B}
    h = {}
    T = {}
    c = {}
    p_price = {}
    for rec in table['records']:
        vals = rec['values']
        equip = vals.get('Equipment / Cost', '').strip()
        if equip in E_A or equip in E_B:
            cost_str = vals.get('Equipment Cost at Full Load (yuan)', '').strip()
            time_str = vals.get('Available Equipment Operating Time', '').strip()
            if cost_str == '' or time_str == '':
                raise ValueError(f'Missing cost or time for equipment {equip}')
            h[equip] = float(cost_str)
            T[equip] = float(time_str)
            for p in P:
                prod_col = f'Product {p}'
                val_str = vals.get(prod_col, '').strip()
                if val_str != '':
                    val = float(val_str)
                    if equip in E_A:
                        tA[p, equip] = val
                    else:
                        tB[p, equip] = val
        elif equip == 'Raw Material Cost (yuan/unit)':
            for p in P:
                prod_col = f'Product {p}'
                val_str = vals.get(prod_col, '').strip()
                if val_str == '':
                    raise ValueError(f'Missing raw material cost for product {p}')
                c[p] = float(val_str)
        elif equip == 'Unit Price (yuan/unit)':
            for p in P:
                prod_col = f'Product {p}'
                val_str = vals.get(prod_col, '').strip()
                if val_str == '':
                    raise ValueError(f'Missing unit price for product {p}')
                p_price[p] = float(val_str)
    for e in E_A + E_B:
        if e not in h or e not in T:
            raise ValueError(f'Missing cost or time for equipment {e}')
    for p in P:
        if p not in c or p not in p_price:
            raise ValueError(f'Missing cost or price for product {p}')
    m = gurobipy.Model()
    m.setParam('MIPGap', 1e-06)
    x = m.addVars(P, lb=0, vtype=GRB.CONTINUOUS, name='')
    a = m.addVars([(p, e) for p in P for e in E_A], lb=0, vtype=GRB.CONTINUOUS, name='')
    b = m.addVars([(p, e) for p in P for e in E_B], lb=0, vtype=GRB.CONTINUOUS, name='')
    for p in P:
        for e in E_A:
            if tA[p, e] is None:
                m.addConstr(a[p, e] == 0)
        for e in E_B:
            if tB[p, e] is None:
                m.addConstr(b[p, e] == 0)
    for p in P:
        m.addConstr(gurobipy.quicksum((a[p, e] for e in E_A)) == x[p])
        m.addConstr(gurobipy.quicksum((b[p, e] for e in E_B)) == x[p])
    for e in E_A:
        expr = gurobipy.quicksum((a[p, e] * tA[p, e] for p in P if tA[p, e] is not None))
        m.addConstr(expr <= T[e])
    for e in E_B:
        expr = gurobipy.quicksum((b[p, e] * tB[p, e] for p in P if tB[p, e] is not None))
        m.addConstr(expr <= T[e])
    profit = gurobipy.quicksum(((p_price[p] - c[p]) * x[p] for p in P))
    cost_A = gurobipy.quicksum((h[e] * gurobipy.quicksum((a[p, e] * tA[p, e] for p in P if tA[p, e] is not None)) for e in E_A))
    cost_B = gurobipy.quicksum((h[e] * gurobipy.quicksum((b[p, e] * tB[p, e] for p in P if tB[p, e] is not None)) for e in E_B))
    m.setObjective(profit - cost_A - cost_B, GRB.MAXIMIZE)
    m.optimize()
    return m
m = solve_problem()