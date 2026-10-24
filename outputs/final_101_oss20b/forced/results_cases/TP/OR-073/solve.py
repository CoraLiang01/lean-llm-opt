CSVQA_DATA = {'query': 'A factory produces three products, I, II, and III. Each product goes through two processing procedures, A and B. The factory has two types of equipment, A1 and A2, to complete procedure A, and three types of equipment, B1, B2, and B3, to complete procedure B. Product I can be processed on either type of A equipment or any type of B equipment. Product II can be processed on any type of A equipment, but when completing procedure B, it can only be processed on B1 equipment. Product III can only be processed on A2 and B2 equipment. Given the processing time, raw material cost, product selling price, available equipment operating time, and equipment cost at full load for each type of equipment, as shown in 43.csv, determine the optimal production plan to maximize profit. All production quantities should be continuous.', 'relationships': [], 'tables': [{'columns': ['Equipment / Cost', 'Product I', 'Product II', 'Product III', 'Available Equipment Operating Time', 'Equipment Cost at Full Load (yuan)'], 'file_name': '43.csv', 'records': [{'data_row_number': 0, 'values': {'Available Equipment Operating Time': '6000', 'Equipment / Cost': 'A1', 'Equipment Cost at Full Load (yuan)': '300', 'Product I': '5', 'Product II': '10', 'Product III': ''}}, {'data_row_number': 1, 'values': {'Available Equipment Operating Time': '10000', 'Equipment / Cost': 'A2', 'Equipment Cost at Full Load (yuan)': '321', 'Product I': '7', 'Product II': '9', 'Product III': '12'}}, {'data_row_number': 2, 'values': {'Available Equipment Operating Time': '8000', 'Equipment / Cost': 'A3', 'Equipment Cost at Full Load (yuan)': '203', 'Product I': '6', 'Product II': '11', 'Product III': '2'}}, {'data_row_number': 3, 'values': {'Available Equipment Operating Time': '4000', 'Equipment / Cost': 'B1', 'Equipment Cost at Full Load (yuan)': '250', 'Product I': '6', 'Product II': '8', 'Product III': ''}}, {'data_row_number': 4, 'values': {'Available Equipment Operating Time': '7000', 'Equipment / Cost': 'B2', 'Equipment Cost at Full Load (yuan)': '783', 'Product I': '4', 'Product II': '', 'Product III': '11'}}, {'data_row_number': 5, 'values': {'Available Equipment Operating Time': '4000', 'Equipment / Cost': 'B3', 'Equipment Cost at Full Load (yuan)': '200', 'Product I': '7', 'Product II': '', 'Product III': ''}}, {'data_row_number': 6, 'values': {'Available Equipment Operating Time': '5000', 'Equipment / Cost': 'B4', 'Equipment Cost at Full Load (yuan)': '300', 'Product I': '3', 'Product II': '5', 'Product III': '8'}}, {'data_row_number': 7, 'values': {'Available Equipment Operating Time': '', 'Equipment / Cost': 'Raw Material Cost (yuan/unit)', 'Equipment Cost at Full Load (yuan)': '', 'Product I': '0.25', 'Product II': '0.35', 'Product III': '0.5'}}, {'data_row_number': 8, 'values': {'Available Equipment Operating Time': '', 'Equipment / Cost': 'Unit Price (yuan/unit)', 'Equipment Cost at Full Load (yuan)': '', 'Product I': '1.25', 'Product II': '2', 'Product III': '2.8'}}], 'table_id': 'file_0_view_0'}]}
import gurobipy
from gurobipy import GRB

def solve_problem():
    table = next((t for t in CSVQA_DATA['tables'] if t['table_id'] == 'file_0_view_0'))
    records = table['records']
    product_ids = ['I', 'II', 'III']
    product_columns = ['Product I', 'Product II', 'Product III']
    product_col_map = dict(zip(product_ids, product_columns))
    a = {}
    c = {}
    t = {}
    r = {}
    s = {}
    allowed_pairs = set()
    equipment_names = []
    for rec in records:
        values = rec['values']
        equip = values['Equipment / Cost']
        if equip == 'Raw Material Cost (yuan)':
            for p in product_ids:
                val = values[product_col_map[p]]
                if val != '':
                    r[p] = float(val)
            continue
        if equip == 'Unit Price (yuan/unit)':
            for p in product_ids:
                val = values[product_col_map[p]]
                if val != '':
                    s[p] = float(val)
            continue
        a_val = values['Available Equipment Operating Time']
        c_val = values['Equipment Cost at Full Load (yuan)']
        if a_val == '' or c_val == '':
            continue
        a[equip] = float(a_val)
        c[equip] = float(c_val)
        equipment_names.append(equip)
        for p in product_ids:
            val = values[product_col_map[p]]
            if val != '':
                t[p, equip] = float(val)
                allowed_pairs.add((p, equip))
    A = [e for e in equipment_names if e.startswith('A')]
    B = [e for e in equipment_names if e.startswith('B')]
    E = A + B
    for p in product_ids:
        if p not in r:
            raise ValueError(f'Missing raw material cost for product {p}')
        if p not in s:
            raise ValueError(f'Missing unit price for product {p}')
    for e in E:
        if e not in a:
            raise ValueError(f'Missing available time for equipment {e}')
        if e not in c:
            raise ValueError(f'Missing cost for equipment {e}')
    for pair in allowed_pairs:
        if pair not in t:
            raise ValueError(f'Missing processing time for pair {pair}')
    model = gurobipy.Model()
    model.setParam('MIPGap', 1e-06)
    x = model.addVars([(p, e) for p, e in allowed_pairs], lb=0, vtype=GRB.CONTINUOUS, name='')
    sumA = {}
    for p in product_ids:
        sumA[p] = gurobipy.quicksum((x[p, e] for e in A if (p, e) in x))
    sum_t = {}
    for e in E:
        sum_t[e] = gurobipy.quicksum((t[p, e] * x[p, e] for p in product_ids if (p, e) in x))
    obj = gurobipy.quicksum(((s[p] - r[p]) * sumA[p] for p in product_ids)) - gurobipy.quicksum((c[e] * sum_t[e] for e in E))
    model.setObjective(obj, GRB.MAXIMIZE)
    model.addConstrs((gurobipy.quicksum((x[p, e] for e in A if (p, e) in x)) == gurobipy.quicksum((x[p, e] for e in B if (p, e) in x)) for p in product_ids), name='')
    model.addConstrs((gurobipy.quicksum((t[p, e] * x[p, e] for p in product_ids if (p, e) in x)) <= a[e] for e in E), name='')
    model.optimize()
    if model.Status == GRB.OPTIMAL:
        print(f'ObjVal: {model.ObjVal}')
        for var in model.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {model.Status}')
    return model
m = solve_problem()
