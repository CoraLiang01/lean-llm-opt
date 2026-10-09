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
             'role': 'equipment and product processing data',
             'table_id': 'file_0_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import pandas as pd
CSVQA_FRAMES = {t["table_id"]: pd.DataFrame([r["values"] for r in t["records"]], columns=t["columns"], index=[r["source_row"] for r in t["records"]]) for t in CSVQA_DATA["tables"]}
import gurobipy as gp
import pandas as pd

def solve_problem(CSVQA_FRAMES):
    frame = CSVQA_FRAMES['file_0_view_0']
    products = ['Product I', 'Product II', 'Product III']
    products_short = {'Product I': 'I', 'Product II': 'II', 'Product III': 'III'}
    P = ['I', 'II', 'III']
    A = ['A1', 'A2']
    B = ['B1', 'B2', 'B3']
    A_p = {'I': ['A1', 'A2'], 'II': ['A1', 'A2'], 'III': ['A2']}
    B_p = {'I': ['B1', 'B2', 'B3'], 'II': ['B1'], 'III': ['B2']}
    equip_row = {}
    for (idx, row) in frame.iterrows():
        eq = row['Equipment / Cost']
        if eq in A or eq in B:
            equip_row[eq] = idx
    t_a_p = {}
    t_b_p = {}
    for a in A:
        row = frame.loc[equip_row[a]]
        for p in P:
            prod_col = f'Product {p}'
            val = row[prod_col]
            if val != '':
                t_a_p[a, p] = float(val)
    for b in B:
        row = frame.loc[equip_row[b]]
        for p in P:
            prod_col = f'Product {p}'
            val = row[prod_col]
            if val != '':
                t_b_p[b, p] = float(val)
    T_a = {}
    T_b = {}
    for a in A:
        row = frame.loc[equip_row[a]]
        T_a[a] = float(row['Available Equipment Operating Time'])
    for b in B:
        row = frame.loc[equip_row[b]]
        T_b[b] = float(row['Available Equipment Operating Time'])
    C_a = {}
    C_b = {}
    for a in A:
        row = frame.loc[equip_row[a]]
        C_a[a] = float(row['Equipment Cost at Full Load (yuan)'])
    for b in B:
        row = frame.loc[equip_row[b]]
        C_b[b] = float(row['Equipment Cost at Full Load (yuan)'])
    rmc_row = frame[frame['Equipment / Cost'] == 'Raw Material Cost (yuan/unit)'].index[0]
    sp_row = frame[frame['Equipment / Cost'] == 'Unit Price (yuan/unit)'].index[0]
    rmc_p = {}
    sp_p = {}
    for p in P:
        prod_col = f'Product {p}'
        rmc_p[p] = float(frame.loc[rmc_row][prod_col])
        sp_p[p] = float(frame.loc[sp_row][prod_col])
    m = gp.Model('FactoryProduction')
    m.Params.MIPGap = 0.0001
    x_vars = m.addVars(P, lb=0, name='')
    y_keys = []
    for p in P:
        for a in A_p[p]:
            y_keys.append((a, p))
    y_vars = m.addVars(y_keys, lb=0, name='')
    z_keys = []
    for p in P:
        for b in B_p[p]:
            z_keys.append((b, p))
    z_vars = m.addVars(z_keys, lb=0, name='')
    profit_expr = gp.quicksum(((sp_p[p] - rmc_p[p]) * x_vars[p] for p in P))
    equipA_cost_expr = gp.quicksum((C_a[a] * gp.quicksum((t_a_p[a, p] * y_vars[a, p] for p in P if (a, p) in y_vars)) / T_a[a] for a in A))
    equipB_cost_expr = gp.quicksum((C_b[b] * gp.quicksum((t_b_p[b, p] * z_vars[b, p] for p in P if (b, p) in z_vars)) / T_b[b] for b in B))
    m.setObjective(profit_expr - equipA_cost_expr - equipB_cost_expr, gp.GRB.MAXIMIZE)
    for p in P:
        m.addConstr(x_vars[p] == gp.quicksum((y_vars[a, p] for a in A_p[p])), name=f'assignA_{p}')
    for p in P:
        m.addConstr(x_vars[p] == gp.quicksum((z_vars[b, p] for b in B_p[p])), name=f'assignB_{p}')
    for a in A:
        m.addConstr(gp.quicksum((t_a_p[a, p] * y_vars[a, p] for p in P if (a, p) in y_vars)) <= T_a[a], name=f'timeA_{a}')
    for b in B:
        m.addConstr(gp.quicksum((t_b_p[b, p] * z_vars[b, p] for p in P if (b, p) in z_vars)) <= T_b[b], name=f'timeB_{b}')
    m.optimize()
    return m
m = solve_problem(CSVQA_FRAMES)