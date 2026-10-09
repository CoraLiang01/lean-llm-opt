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
             'role': 'equipment-product processing and cost data',
             'table_id': 'file_0_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import pandas as pd
CSVQA_FRAMES = {t["table_id"]: pd.DataFrame([r["values"] for r in t["records"]], columns=t["columns"], index=[r["source_row"] for r in t["records"]]) for t in CSVQA_DATA["tables"]}
import gurobipy as gp
import pandas as pd

def solve_problem(CSVQA_FRAMES):
    frame = CSVQA_FRAMES['file_0_view_0']
    products = ['Product I', 'Product II', 'Product III']
    A_equip = ['A1', 'A2']
    B_equip = ['B1', 'B2', 'B3']
    prod_short = {'Product I': 'I', 'Product II': 'II', 'Product III': 'III'}
    prod_long = {v: k for (k, v) in prod_short.items()}
    raw_mat_row = frame[frame['Equipment / Cost'].str.casefold() == 'raw material cost (yuan/unit)'.casefold()].iloc[0]
    price_row = frame[frame['Equipment / Cost'].str.casefold() == 'unit price (yuan/unit)'.casefold()].iloc[0]
    c_p = {prod_short[p]: float(raw_mat_row[p]) for p in products}
    s_p = {prod_short[p]: float(price_row[p]) for p in products}
    t_a_p = {}
    T_a = {}
    C_a = {}
    t_b_p = {}
    T_b = {}
    C_b = {}
    eligible_x = []
    eligible_y = []
    for a in A_equip:
        row = frame[frame['Equipment / Cost'] == a]
        if row.empty:
            raise ValueError(f'Missing equipment row for {a}')
        row = row.iloc[0]
        T_a[a] = float(row['Available Equipment Operating Time'])
        C_a[a] = float(row['Equipment Cost at Full Load (yuan)'])
        for p in products:
            val = row[p]
            if val != '':
                t_a_p[a, prod_short[p]] = float(val)
                eligible_x.append((a, prod_short[p]))
    for b in B_equip:
        row = frame[frame['Equipment / Cost'] == b]
        if row.empty:
            raise ValueError(f'Missing equipment row for {b}')
        row = row.iloc[0]
        T_b[b] = float(row['Available Equipment Operating Time'])
        C_b[b] = float(row['Equipment Cost at Full Load (yuan)'])
        for p in products:
            val = row[p]
            if val != '':
                t_b_p[b, prod_short[p]] = float(val)
                eligible_y.append((b, prod_short[p]))
    m = gp.Model('FactoryProduction')
    m.Params.MIPGap = 0.0001
    x_vars = m.addVars(eligible_x, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
    y_vars = m.addVars(eligible_y, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
    for a in A_equip:
        terms = []
        for (aa, pp) in eligible_x:
            if aa == a:
                terms.append(t_a_p[a, pp] * x_vars[a, pp])
        m.addConstr(gp.quicksum(terms) <= T_a[a])
    for b in B_equip:
        terms = []
        for (bb, pp) in eligible_y:
            if bb == b:
                terms.append(t_b_p[b, pp] * y_vars[b, pp])
        m.addConstr(gp.quicksum(terms) <= T_b[b])
    m.addConstr(gp.quicksum((x_vars[a, 'I'] for a in A_equip if (a, 'I') in x_vars)) == gp.quicksum((y_vars[b, 'I'] for b in B_equip if (b, 'I') in y_vars)))
    m.addConstr(gp.quicksum((x_vars[a, 'II'] for a in A_equip if (a, 'II') in x_vars)) == y_vars['B1', 'II'])
    m.addConstr(x_vars['A2', 'III'] == y_vars['B2', 'III'])
    q = {}
    q['I'] = gp.quicksum((x_vars[a, 'I'] for a in A_equip if (a, 'I') in x_vars))
    q['II'] = gp.quicksum((x_vars[a, 'II'] for a in A_equip if (a, 'II') in x_vars))
    q['III'] = x_vars['A2', 'III']
    equip_cost_A = gp.quicksum((C_a[a] / T_a[a] * gp.quicksum((t_a_p[a, p] * x_vars[a, p] for (aa, p) in eligible_x if aa == a)) for a in A_equip))
    equip_cost_B = gp.quicksum((C_b[b] / T_b[b] * gp.quicksum((t_b_p[b, p] * y_vars[b, p] for (bb, p) in eligible_y if bb == b)) for b in B_equip))
    m.setObjective(gp.quicksum((s_p[p] * q[p] for p in ['I', 'II', 'III'])) - gp.quicksum((c_p[p] * q[p] for p in ['I', 'II', 'III'])) - equip_cost_A - equip_cost_B, gp.GRB.MAXIMIZE)
    m.optimize()
    return m
m = solve_problem(CSVQA_FRAMES)