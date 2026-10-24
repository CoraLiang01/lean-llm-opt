CSVQA_DATA = {'query': 'A factory produces three products, I, II, and III. Each product goes through two processing procedures, A and B. The factory has two types of equipment, A1 and A2, to complete procedure A, and three types of equipment, B1, B2, and B3, to complete procedure B. Product I can be processed on either type of A equipment or any type of B equipment. Product II can be processed on any type of A equipment, but when completing procedure B, it can only be processed on B1 equipment. Product III can only be processed on A2 and B2 equipment. Given the processing time, raw material cost, product selling price, available equipment operating time, and equipment cost at full load for each type of equipment, as shown in 43.csv, determine the optimal production plan to maximize profit. All production quantities should be continuous.', 'relationships': [], 'tables': [{'columns': ['Equipment / Cost', 'Product I', 'Product II', 'Product III', 'Available Equipment Operating Time', 'Equipment Cost at Full Load (yuan)'], 'file_name': '43.csv', 'records': [{'data_row_number': 0, 'values': {'Available Equipment Operating Time': '6000', 'Equipment / Cost': 'A1', 'Equipment Cost at Full Load (yuan)': '300', 'Product I': '5', 'Product II': '10', 'Product III': ''}}, {'data_row_number': 1, 'values': {'Available Equipment Operating Time': '10000', 'Equipment / Cost': 'A2', 'Equipment Cost at Full Load (yuan)': '321', 'Product I': '7', 'Product II': '9', 'Product III': '12'}}, {'data_row_number': 2, 'values': {'Available Equipment Operating Time': '8000', 'Equipment / Cost': 'A3', 'Equipment Cost at Full Load (yuan)': '203', 'Product I': '6', 'Product II': '11', 'Product III': '2'}}, {'data_row_number': 3, 'values': {'Available Equipment Operating Time': '4000', 'Equipment / Cost': 'B1', 'Equipment Cost at Full Load (yuan)': '250', 'Product I': '6', 'Product II': '8', 'Product III': ''}}, {'data_row_number': 4, 'values': {'Available Equipment Operating Time': '7000', 'Equipment / Cost': 'B2', 'Equipment Cost at Full Load (yuan)': '783', 'Product I': '4', 'Product II': '', 'Product III': '11'}}, {'data_row_number': 5, 'values': {'Available Equipment Operating Time': '4000', 'Equipment / Cost': 'B3', 'Equipment Cost at Full Load (yuan)': '200', 'Product I': '7', 'Product II': '', 'Product III': ''}}, {'data_row_number': 6, 'values': {'Available Equipment Operating Time': '5000', 'Equipment / Cost': 'B4', 'Equipment Cost at Full Load (yuan)': '300', 'Product I': '3', 'Product II': '5', 'Product III': '8'}}, {'data_row_number': 7, 'values': {'Available Equipment Operating Time': '', 'Equipment / Cost': 'Raw Material Cost (yuan/unit)', 'Equipment Cost at Full Load (yuan)': '', 'Product I': '0.25', 'Product II': '0.35', 'Product III': '0.5'}}, {'data_row_number': 8, 'values': {'Available Equipment Operating Time': '', 'Equipment / Cost': 'Unit Price (yuan/unit)', 'Equipment Cost at Full Load (yuan)': '', 'Product I': '1.25', 'Product II': '2', 'Product III': '2.8'}}], 'table_id': 'file_0_view_0'}]}
import gurobipy
from gurobipy import GRB

def solve_problem():
    table = next((t for t in CSVQA_DATA['tables'] if t['table_id'] == 'file_0_view_0'))
    records = table['records']
    product_cols = ['Product I', 'Product II', 'Product III']
    product_map = {'Product I': 'I', 'Product II': 'II', 'Product III': 'III'}
    P = ['I', 'II', 'III']
    a_e = {}
    c_e = {}
    t_e_p = {}
    r_p = {}
    s_p = {}
    for rec in records:
        values = rec['values']
        equip = values['Equipment / Cost'].strip()
        if equip == 'Raw Material Cost (yuan/unit)':
            for col in product_cols:
                key = product_map[col]
                val = values[col]
                r_p[key] = float(val) if val != '' else 0.0
        elif equip == 'Unit Price (yuan/unit)':
            for col in product_cols:
                key = product_map[col]
                val = values[col]
                s_p[key] = float(val) if val != '' else 0.0
        else:
            a_val = values['Available Equipment Operating Time']
            c_val = values['Equipment Cost at Full Load (yuan)']
            a_e[equip] = float(a_val) if a_val != '' else 0.0
            c_e[equip] = float(c_val) if c_val != '' else 0.0
            t_e_p[equip] = {}
            for col in product_cols:
                key = product_map[col]
                val = values[col]
                t_e_p[equip][key] = float(val) if val != '' else 0.0
    E_A = [e for e in a_e if e.startswith('A')]
    E_B = [e for e in a_e if e.startswith('B')]
    for e in E_A + E_B:
        if e not in a_e or e not in c_e:
            raise ValueError(f'Missing operating time or cost for equipment {e}')
        if e not in t_e_p:
            raise ValueError(f'Missing processing times for equipment {e}')
        for p in P:
            if p not in t_e_p[e]:
                t_e_p[e][p] = 0.0
    for p in P:
        if p not in r_p or p not in s_p:
            raise ValueError(f'Missing raw material cost or selling price for product {p}')
    m = gurobipy.Model()
    m.setParam('MIPGap', 1e-06)
    Q = m.addVars(P, lb=0, vtype=GRB.CONTINUOUS, name='')
    x = m.addVars([(e, p) for e in E_A for p in P], lb=0, vtype=GRB.CONTINUOUS, name='')
    y = m.addVars([(e, p) for e in E_B for p in P], lb=0, vtype=GRB.CONTINUOUS, name='')
    for p in P:
        m.addConstr(gurobipy.quicksum((x[e, p] for e in E_A)) == Q[p], name=f'procA_{p}')
        m.addConstr(gurobipy.quicksum((y[e, p] for e in E_B)) == Q[p], name=f'procB_{p}')
    for e in E_A:
        m.addConstr(gurobipy.quicksum((t_e_p[e][p] * x[e, p] for p in P)) <= a_e[e], name=f'timeA_{e}')
    for e in E_B:
        m.addConstr(gurobipy.quicksum((t_e_p[e][p] * y[e, p] for p in P)) <= a_e[e], name=f'timeB_{e}')
    for e in E_A:
        for p in P:
            if t_e_p[e][p] == 0.0:
                m.addConstr(x[e, p] == 0, name=f'zeroA_{e}_{p}')
    for e in E_B:
        for p in P:
            if t_e_p[e][p] == 0.0:
                m.addConstr(y[e, p] == 0, name=f'zeroB_{e}_{p}')
    profit_terms = gurobipy.quicksum(((s_p[p] - r_p[p]) * Q[p] for p in P))
    cost_A = gurobipy.quicksum((c_e[e] * gurobipy.quicksum((t_e_p[e][p] * x[e, p] for p in P)) for e in E_A))
    cost_B = gurobipy.quicksum((c_e[e] * gurobipy.quicksum((t_e_p[e][p] * y[e, p] for p in P)) for e in E_B))
    m.setObjective(profit_terms - cost_A - cost_B, GRB.MAXIMIZE)
    m.optimize()
    return m
m = solve_problem()
