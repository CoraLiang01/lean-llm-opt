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
 'route': 'NRM',
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
    table = CSVQA_DATA['tables'][0]
    records = table['records']
    product_cols = [col for col in records[0]['values'].keys() if col.startswith('Product ')]
    P = set(product_cols)
    A = set()
    B = set()
    t_pa = {}
    f_pa = {}
    t_pb = {}
    f_pb = {}
    T_a = {}
    T_b = {}
    c_a = {}
    c_b = {}
    r_p = {}
    p_p = {}

    def to_float(val):
        return float(val) if val != '' else None
    for rec in records:
        equip = rec['values']['Equipment / Cost']
        if equip in ('Raw Material Cost (yuan/unit)', 'Unit Price (yuan/unit)'):
            continue
        name = equip.strip()
        avail_time = to_float(rec['values']['Available Equipment Operating Time'])
        cost_per_hour = to_float(rec['values']['Equipment Cost at Full Load (yuan)'])
        if name.startswith('A'):
            A.add(name)
            T_a[name] = avail_time if avail_time is not None else 0.0
            c_a[name] = cost_per_hour if cost_per_hour is not None else 0.0
            for p in P:
                t_val = to_float(rec['values'][p])
                if t_val is not None:
                    t_pa[p, name] = t_val
                    f_pa[p, name] = 1
                else:
                    t_pa[p, name] = 0.0
                    f_pa[p, name] = 0
        elif name.startswith('B'):
            B.add(name)
            T_b[name] = avail_time if avail_time is not None else 0.0
            c_b[name] = cost_per_hour if cost_per_hour is not None else 0.0
            for p in P:
                t_val = to_float(rec['values'][p])
                if t_val is not None:
                    t_pb[p, name] = t_val
                    f_pb[p, name] = 1
                else:
                    t_pb[p, name] = 0.0
                    f_pb[p, name] = 0
    for rec in records:
        if rec['values']['Equipment / Cost'] == 'Raw Material Cost (yuan/unit)':
            for p in P:
                val = to_float(rec['values'][p])
                r_p[p] = val if val is not None else 0.0
            break
    for rec in records:
        if rec['values']['Equipment / Cost'] == 'Unit Price (yuan/unit)':
            for p in P:
                val = to_float(rec['values'][p])
                p_p[p] = val if val is not None else 0.0
            break
    model = gurobipy.Model()
    model.setParam('MIPGap', 1e-06)
    x_pa = model.addVars([(p, a) for p in P for a in A], lb=0, vtype=GRB.CONTINUOUS, name='')
    x_pb = model.addVars([(p, b) for p in P for b in B], lb=0, vtype=GRB.CONTINUOUS, name='')
    y_p = model.addVars(P, lb=0, vtype=GRB.CONTINUOUS, name='')
    model.addConstrs((gurobipy.quicksum((x_pa[p, a] for a in A)) == y_p[p] for p in P), name='')
    model.addConstrs((gurobipy.quicksum((x_pb[p, b] for b in B)) == y_p[p] for p in P), name='')
    model.addConstrs((gurobipy.quicksum((t_pa[p, a] * x_pa[p, a] for p in P)) <= T_a[a] for a in A), name='')
    model.addConstrs((gurobipy.quicksum((t_pb[p, b] * x_pb[p, b] for p in P)) <= T_b[b] for b in B), name='')
    M = 1000000.0
    model.addConstrs((x_pa[p, a] <= M * f_pa[p, a] for p in P for a in A), name='')
    model.addConstrs((x_pb[p, b] <= M * f_pb[p, b] for p in P for b in B), name='')
    profit_terms = gurobipy.quicksum(((p_p[p] - r_p[p]) * y_p[p] for p in P))
    cost_terms_a = gurobipy.quicksum((c_a[a] * t_pa[p, a] * x_pa[p, a] for p in P for a in A))
    cost_terms_b = gurobipy.quicksum((c_b[b] * t_pb[p, b] * x_pb[p, b] for p in P for b in B))
    model.setObjective(profit_terms - cost_terms_a - cost_terms_b, GRB.MAXIMIZE)
    model.optimize()
    return model
m = solve_problem()