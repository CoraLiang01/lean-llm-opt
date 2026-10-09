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
 'route': 'RA',
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
             'role': 'equipment-product processing and cost matrix',
             'table_id': 'file_0_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import pandas as pd
CSVQA_FRAMES = {t["table_id"]: pd.DataFrame([r["values"] for r in t["records"]], columns=t["columns"], index=[r["source_row"] for r in t["records"]]) for t in CSVQA_DATA["tables"]}
import gurobipy as gp
from gurobipy import GRB
import pandas as pd

def solve_problem(CSVQA_FRAMES):
    df = CSVQA_FRAMES['file_0_view_0']
    products = ['Product I', 'Product II', 'Product III']
    A_equipment = ['A1', 'A2']
    B_equipment = ['B1', 'B2', 'B3']
    A_p = {'Product I': ['A1', 'A2'], 'Product II': ['A1', 'A2'], 'Product III': ['A2']}
    B_p = {'Product I': ['B1', 'B2', 'B3'], 'Product II': ['B1'], 'Product III': ['B2']}
    P_e_A = {e: [] for e in A_equipment}
    for p in products:
        for e in A_p[p]:
            P_e_A[e].append(p)
    P_e_B = {e: [] for e in B_equipment}
    for p in products:
        for e in B_p[p]:
            P_e_B[e].append(p)
    t_A = {}
    t_B = {}
    T_e = {}
    C_e = {}
    for (idx, row) in df.iterrows():
        eq = row['Equipment / Cost']
        if eq in A_equipment:
            for p in products:
                val = row[p]
                if val != '':
                    t_A[eq, p] = float(val)
            T_e[eq] = float(row['Available Equipment Operating Time'])
            C_e[eq] = float(row['Equipment Cost at Full Load (yuan)'])
        elif eq in B_equipment:
            for p in products:
                val = row[p]
                if val != '':
                    t_B[eq, p] = float(val)
            T_e[eq] = float(row['Available Equipment Operating Time'])
            C_e[eq] = float(row['Equipment Cost at Full Load (yuan)'])
    raw_row = df[df['Equipment / Cost'] == 'Raw Material Cost (yuan/unit)'].iloc[0]
    price_row = df[df['Equipment / Cost'] == 'Unit Price (yuan/unit)'].iloc[0]
    c_raw = {p: float(raw_row[p]) for p in products}
    r_p = {p: float(price_row[p]) for p in products}
    m = gp.Model('factory_production')
    x_vars = m.addVars(products, lb=0, vtype=GRB.CONTINUOUS, name='')
    yA_keys = [(e, p) for e in A_equipment for p in products if e in A_p[p]]
    yB_keys = [(e, p) for e in B_equipment for p in products if e in B_p[p]]
    yA_vars = m.addVars(yA_keys, lb=0, vtype=GRB.CONTINUOUS, name='')
    yB_vars = m.addVars(yB_keys, lb=0, vtype=GRB.CONTINUOUS, name='')
    sales_minus_raw = gp.quicksum(((r_p[p] - c_raw[p]) * x_vars[p] for p in products))
    equip_cost_A = gp.quicksum((C_e[e] / T_e[e] * gp.quicksum((t_A[e, p] * yA_vars[e, p] for p in P_e_A[e])) for e in A_equipment))
    equip_cost_B = gp.quicksum((C_e[e] / T_e[e] * gp.quicksum((t_B[e, p] * yB_vars[e, p] for p in P_e_B[e])) for e in B_equipment))
    m.setObjective(sales_minus_raw - equip_cost_A - equip_cost_B, GRB.MAXIMIZE)
    for p in products:
        m.addConstr(x_vars[p] == gp.quicksum((yA_vars[e, p] for e in A_p[p])), name=f'consist_A_{p}')
        m.addConstr(x_vars[p] == gp.quicksum((yB_vars[e, p] for e in B_p[p])), name=f'consist_B_{p}')
    for e in A_equipment:
        m.addConstr(gp.quicksum((t_A[e, p] * yA_vars[e, p] for p in P_e_A[e])) <= T_e[e], name=f'time_A_{e}')
    for e in B_equipment:
        m.addConstr(gp.quicksum((t_B[e, p] * yB_vars[e, p] for p in P_e_B[e])) <= T_e[e], name=f'time_B_{e}')
    m.setParam('MIPGap', 0.0001)
    m.optimize()
    return m
m = solve_problem(CSVQA_FRAMES)
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')