CSVQA_DATA = {'query': 'A factory produces three products, I, II, and III. Each product goes through two processing procedures, A and B. The factory has two types of equipment, A1 and A2, to complete procedure A, and three types of equipment, B1, B2, and B3, to complete procedure B. Product I can be processed on either type of A equipment or any type of B equipment. Product II can be processed on any type of A equipment, but when completing procedure B, it can only be processed on B1 equipment. Product III can only be processed on A2 and B2 equipment. Given the processing time, raw material cost, product selling price, available equipment operating time, and equipment cost at full load for each type of equipment, as shown in 43.csv, determine the optimal production plan to maximize profit. All production quantities should be continuous.', 'relationships': [], 'tables': [{'columns': ['Equipment / Cost', 'Product I', 'Product II', 'Product III', 'Available Equipment Operating Time', 'Equipment Cost at Full Load (yuan)'], 'file_name': '43.csv', 'records': [{'data_row_number': 0, 'values': {'Available Equipment Operating Time': '6000', 'Equipment / Cost': 'A1', 'Equipment Cost at Full Load (yuan)': '300', 'Product I': '5', 'Product II': '10', 'Product III': ''}}, {'data_row_number': 1, 'values': {'Available Equipment Operating Time': '10000', 'Equipment / Cost': 'A2', 'Equipment Cost at Full Load (yuan)': '321', 'Product I': '7', 'Product II': '9', 'Product III': '12'}}, {'data_row_number': 2, 'values': {'Available Equipment Operating Time': '8000', 'Equipment / Cost': 'A3', 'Equipment Cost at Full Load (yuan)': '203', 'Product I': '6', 'Product II': '11', 'Product III': '2'}}, {'data_row_number': 3, 'values': {'Available Equipment Operating Time': '4000', 'Equipment / Cost': 'B1', 'Equipment Cost at Full Load (yuan)': '250', 'Product I': '6', 'Product II': '8', 'Product III': ''}}], 'table_id': 'file_0_view_0'}, {'columns': ['Equipment / Cost', 'Product I', 'Product II', 'Product III', 'Available Equipment Operating Time', 'Equipment Cost at Full Load (yuan)'], 'file_name': '43.csv', 'records': [{'data_row_number': 0, 'values': {'Available Equipment Operating Time': '6000', 'Equipment / Cost': 'A1', 'Equipment Cost at Full Load (yuan)': '300', 'Product I': '5', 'Product II': '10', 'Product III': ''}}, {'data_row_number': 1, 'values': {'Available Equipment Operating Time': '10000', 'Equipment / Cost': 'A2', 'Equipment Cost at Full Load (yuan)': '321', 'Product I': '7', 'Product II': '9', 'Product III': '12'}}, {'data_row_number': 2, 'values': {'Available Equipment Operating Time': '8000', 'Equipment / Cost': 'A3', 'Equipment Cost at Full Load (yuan)': '203', 'Product I': '6', 'Product II': '11', 'Product III': '2'}}, {'data_row_number': 3, 'values': {'Available Equipment Operating Time': '4000', 'Equipment / Cost': 'B1', 'Equipment Cost at Full Load (yuan)': '250', 'Product I': '6', 'Product II': '8', 'Product III': ''}}], 'table_id': 'file_0_view_1'}, {'columns': ['Equipment / Cost', 'Product I', 'Product II', 'Product III', 'Available Equipment Operating Time', 'Equipment Cost at Full Load (yuan)'], 'file_name': '43.csv', 'records': [{'data_row_number': 1, 'values': {'Available Equipment Operating Time': '10000', 'Equipment / Cost': 'A2', 'Equipment Cost at Full Load (yuan)': '321', 'Product I': '7', 'Product II': '9', 'Product III': '12'}}, {'data_row_number': 4, 'values': {'Available Equipment Operating Time': '7000', 'Equipment / Cost': 'B2', 'Equipment Cost at Full Load (yuan)': '783', 'Product I': '4', 'Product II': '', 'Product III': '11'}}], 'table_id': 'file_0_view_2'}]}
import gurobipy
from gurobipy import GRB
equipments = set()
products = ['I', 'II', 'III']
t = {}
T = {}
c = {}
for table in CSVQA_DATA['tables']:
    for rec in table['records']:
        values = rec['values']
        e = values.get('Equipment / Cost')
        if not e:
            continue
        equipments.add(e)
        T_val = values.get('Available Equipment Operating Time')
        if T_val not in (None, ''):
            T_val_f = float(T_val)
            if e in T and T[e] != T_val_f:
                raise ValueError(f'Inconsistent T for equipment {e}')
            T[e] = T_val_f
        c_val = values.get('Equipment Cost at Full Load (yuan)')
        if c_val not in (None, ''):
            c_val_f = float(c_val)
            if e in c and c[e] != c_val_f:
                raise ValueError(f'Inconsistent c for equipment {e}')
            c[e] = c_val_f
        if e not in t:
            t[e] = {}
        for col, prod in [('Product I', 'I'), ('Product II', 'II'), ('Product III', 'III')]:
            val = values.get(col)
            if val not in (None, ''):
                val_f = float(val)
                if prod in t[e] and t[e][prod] != val_f:
                    raise ValueError(f'Inconsistent t for equipment {e}, product {prod}')
                t[e][prod] = val_f
pairs = [(e, p) for e in equipments for p in products if p in t.get(e, {})]
m = gurobipy.Model()
m.setParam('MIPGap', 1e-06)
x = m.addVars(pairs, vtype=GRB.CONTINUOUS, name='')
pi = {'I': 0.0, 'II': 0.0, 'III': 0.0}
obj = gurobipy.quicksum((pi[p] * x[e, p] - c[e] * t[e][p] * x[e, p] for e, p in pairs))
m.setObjective(obj, GRB.MAXIMIZE)
for e in equipments:
    expr = gurobipy.quicksum((t[e][p] * x[e, p] for p in products if (e, p) in pairs))
    m.addConstr(expr <= T[e], name=f'time_{e}')
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')
