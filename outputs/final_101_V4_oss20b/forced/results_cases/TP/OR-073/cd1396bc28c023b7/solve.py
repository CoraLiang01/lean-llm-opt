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
 'route': 'TP',
 'tables': [{'columns': ['Equipment / Cost',
                         'Product I',
                         'Product II',
                         'Product III',
                         'Available Equipment Operating Time',
                         'Equipment Cost at Full Load (yuan)'],
             'file_index': 0,
             'file_name': '43.csv',
             'filters': {'conditions': [], 'logic': 'and'},
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
    table = next((t for t in CSVQA_DATA['tables'] if t['table_id'] == 'file_0_view_0'))
    records = table['records']
    P = ['I', 'II', 'III']
    product_cols = {'I': 'Product I', 'II': 'Product II', 'III': 'Product III'}

    def to_float(s):
        return float(s) if s != '' else None
    t_A = {}
    t_B = {}
    a = {}
    c = {}
    r = {}
    p_price = {}
    A_set = set()
    B_set = set()
    for rec in records:
        equip = rec['values']['Equipment / Cost']
        if equip in ('Raw Material Cost (yuan/unit)', 'Unit Price (yuan/unit)'):
            if equip == 'Raw Material Cost (yuan/unit)':
                for p in P:
                    val = rec['values'][product_cols[p]]
                    if val == '':
                        raise ValueError(f'Missing raw material cost for product {p}')
                    r[p] = float(val)
            else:
                for p in P:
                    val = rec['values'][product_cols[p]]
                    if val == '':
                        raise ValueError(f'Missing unit price for product {p}')
                    p_price[p] = float(val)
            continue
        a_e = to_float(rec['values']['Available Equipment Operating Time'])
        c_e = to_float(rec['values']['Equipment Cost at Full Load (yuan)'])
        if a_e is None or c_e is None:
            raise ValueError(f'Missing availability or cost for equipment {equip}')
        a[equip] = a_e
        c[equip] = c_e
        if equip.startswith('A'):
            A_set.add(equip)
            t_A[equip] = {}
        elif equip.startswith('B'):
            B_set.add(equip)
            t_B[equip] = {}
        else:
            raise ValueError(f'Unknown equipment prefix in {equip}')
        for p in P:
            val = to_float(rec['values'][product_cols[p]])
            if val is not None:
                if equip.startswith('A'):
                    t_A[equip][p] = val
                else:
                    t_B[equip][p] = val
    for p in P:
        if p not in r:
            raise ValueError(f'Raw material cost missing for product {p}')
        if p not in p_price:
            raise ValueError(f'Unit price missing for product {p}')
    for e in A_set | B_set:
        if e not in a:
            raise ValueError(f'Available time missing for equipment {e}')
        if e not in c:
            raise ValueError(f'Cost missing for equipment {e}')
    model = gurobipy.Model()
    model.setParam('MIPGap', 1e-06)
    xA = model.addVars([(p, e) for e in A_set for p in P if p in t_A.get(e, {})], lb=0, vtype=GRB.CONTINUOUS, name='')
    xB = model.addVars([(p, e) for e in B_set for p in P if p in t_B.get(e, {})], lb=0, vtype=GRB.CONTINUOUS, name='')
    y = model.addVars(P, lb=0, vtype=GRB.CONTINUOUS, name='')
    model.addConstrs((gurobipy.quicksum((xA[p, e] for e in A_set if (p, e) in xA)) == y[p] for p in P), name='')
    model.addConstrs((gurobipy.quicksum((xB[p, e] for e in B_set if (p, e) in xB)) == y[p] for p in P), name='')
    model.addConstrs((gurobipy.quicksum((t_A[e][p] * xA[p, e] for p in P if (p, e) in xA)) <= a[e] for e in A_set), name='')
    model.addConstrs((gurobipy.quicksum((t_B[e][p] * xB[p, e] for p in P if (p, e) in xB)) <= a[e] for e in B_set), name='')
    profit_unit = {p: p_price[p] - r[p] for p in P}
    obj = gurobipy.quicksum((profit_unit[p] * y[p] for p in P))
    obj -= gurobipy.quicksum((c[e] * gurobipy.quicksum((t_A[e][p] * xA[p, e] for p in P if (p, e) in xA)) for e in A_set))
    obj -= gurobipy.quicksum((c[e] * gurobipy.quicksum((t_B[e][p] * xB[p, e] for p in P if (p, e) in xB)) for e in B_set))
    model.setObjective(obj, GRB.MAXIMIZE)
    model.optimize()
    return model
m = solve_problem()