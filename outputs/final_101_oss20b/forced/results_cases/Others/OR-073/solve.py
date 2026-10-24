CSVQA_DATA = {'query': 'A factory produces three products, I, II, and III. Each product goes through two processing procedures, A and B. The factory has two types of equipment, A1 and A2, to complete procedure A, and three types of equipment, B1, B2, and B3, to complete procedure B. Product I can be processed on either type of A equipment or any type of B equipment. Product II can be processed on any type of A equipment, but when completing procedure B, it can only be processed on B1 equipment. Product III can only be processed on A2 and B2 equipment. Given the processing time, raw material cost, product selling price, available equipment operating time, and equipment cost at full load for each type of equipment, as shown in 43.csv, determine the optimal production plan to maximize profit. All production quantities should be continuous.', 'relationships': [], 'tables': [{'columns': ['Equipment / Cost', 'Product I', 'Product II', 'Product III', 'Available Equipment Operating Time', 'Equipment Cost at Full Load (yuan)'], 'file_name': '43.csv', 'records': [{'data_row_number': 0, 'values': {'Available Equipment Operating Time': '6000', 'Equipment / Cost': 'A1', 'Equipment Cost at Full Load (yuan)': '300', 'Product I': '5', 'Product II': '10', 'Product III': ''}}, {'data_row_number': 1, 'values': {'Available Equipment Operating Time': '10000', 'Equipment / Cost': 'A2', 'Equipment Cost at Full Load (yuan)': '321', 'Product I': '7', 'Product II': '9', 'Product III': '12'}}, {'data_row_number': 2, 'values': {'Available Equipment Operating Time': '8000', 'Equipment / Cost': 'A3', 'Equipment Cost at Full Load (yuan)': '203', 'Product I': '6', 'Product II': '11', 'Product III': '2'}}, {'data_row_number': 3, 'values': {'Available Equipment Operating Time': '4000', 'Equipment / Cost': 'B1', 'Equipment Cost at Full Load (yuan)': '250', 'Product I': '6', 'Product II': '8', 'Product III': ''}}, {'data_row_number': 4, 'values': {'Available Equipment Operating Time': '7000', 'Equipment / Cost': 'B2', 'Equipment Cost at Full Load (yuan)': '783', 'Product I': '4', 'Product II': '', 'Product III': '11'}}, {'data_row_number': 5, 'values': {'Available Equipment Operating Time': '4000', 'Equipment / Cost': 'B3', 'Equipment Cost at Full Load (yuan)': '200', 'Product I': '7', 'Product II': '', 'Product III': ''}}, {'data_row_number': 6, 'values': {'Available Equipment Operating Time': '5000', 'Equipment / Cost': 'B4', 'Equipment Cost at Full Load (yuan)': '300', 'Product I': '3', 'Product II': '5', 'Product III': '8'}}, {'data_row_number': 7, 'values': {'Available Equipment Operating Time': '', 'Equipment / Cost': 'Raw Material Cost (yuan/unit)', 'Equipment Cost at Full Load (yuan)': '', 'Product I': '0.25', 'Product II': '0.35', 'Product III': '0.5'}}, {'data_row_number': 8, 'values': {'Available Equipment Operating Time': '', 'Equipment / Cost': 'Unit Price (yuan/unit)', 'Equipment Cost at Full Load (yuan)': '', 'Product I': '1.25', 'Product II': '2', 'Product III': '2.8'}}], 'table_id': 'file_0_view_0'}]}
import gurobipy
from gurobipy import GRB

def solve_problem():
    data = CSVQA_DATA
    tables = data['tables']
    table = next((t for t in tables if t['table_id'] == 'file_0_view_0'))
    records = table['records']
    products = ['I', 'II', 'III']
    equipment = []
    a_e = {}
    c_e = {}
    t_ep = {}
    r_p = {}
    s_p = {}
    for rec in records:
        equip = rec['values']['Equipment / Cost']
        if equip == 'Raw Material Cost (yuan/unit)':
            for p in products:
                val = rec['values'][f'Product {p}']
                if val == '':
                    raise ValueError(f'Missing raw material cost for product {p}')
                r_p[p] = float(val)
        elif equip == 'Unit Price (yuan/unit)':
            for p in products:
                val = rec['values'][f'Product {p}']
                if val == '':
                    raise ValueError(f'Missing unit price for product {p}')
                s_p[p] = float(val)
        else:
            equipment.append(equip)
            a_e[equip] = float(rec['values']['Available Equipment Operating Time'])
            c_e[equip] = float(rec['values']['Equipment Cost at Full Load (yuan)'])
            for p in products:
                val = rec['values'][f'Product {p}']
                if val != '':
                    t_ep[equip, p] = float(val)
    A = [e for e in equipment if e.startswith('A')]
    B = [e for e in equipment if e.startswith('B')]
    m = gurobipy.Model('ProductionProfit')
    m.setParam('MIPGap', 1e-06)
    x = m.addVars(products, lb=0, vtype=GRB.CONTINUOUS, name='')
    z = m.addVars(t_ep.keys(), lb=0, vtype=GRB.CONTINUOUS, name='')
    for e in equipment:
        expr = gurobipy.quicksum((t_ep[e, p] * z[e, p] for p in products if (e, p) in t_ep))
        m.addConstr(expr <= a_e[e], name=f'Cap_{e}')
    for p in products:
        exprA = gurobipy.quicksum((z[e, p] for e in A if (e, p) in t_ep))
        m.addConstr(exprA == x[p], name=f'ProcA_{p}')
        exprB = gurobipy.quicksum((z[e, p] for e in B if (e, p) in t_ep))
        m.addConstr(exprB == x[p], name=f'ProcB_{p}')
    obj = gurobipy.quicksum(((s_p[p] - r_p[p]) * x[p] for p in products))
    obj -= gurobipy.quicksum((c_e[e] * t_ep[e, p] * z[e, p] for e, p in t_ep))
    m.setObjective(obj, GRB.MAXIMIZE)
    m.optimize()
    return m
m = solve_problem()
if m.status == GRB.OPTIMAL:
    print(m.objVal)
    for var in m.getVars():
        print(var.VarName, var.X)
else:
    print(m.status)
