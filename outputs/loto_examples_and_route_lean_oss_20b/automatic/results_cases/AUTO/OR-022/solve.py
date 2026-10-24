CSVQA_DATA = {'query': 'The department store is hosting a promotional event featuring various top-selling items. Revenue data is available in the ‘Revenue’ column. The retailer aims to maximize total revenue using the initial inventory of products classified under ‘27in’. Inventory levels are detailed in the ‘Initial Inventory’ column. Demand quantities are provided in the ‘Demand’ column and are assumed to be deterministic and known in advance. Decision variables x_i indicate the number of units of each ‘27in’ product i that will be fulfilled.', 'relationships': [], 'tables': [{'columns': ['Product Name', 'Revenue', 'Demand', 'Initial Inventory'], 'file_name': 'Salesorders.csv', 'records': [{'data_row_number': 1, 'values': {'Demand': '12474', 'Initial Inventory': '62440', 'Product Name': '27in 4K Gaming Monitor', 'Revenue': '261.2933'}}, {'data_row_number': 2, 'values': {'Demand': '15057', 'Initial Inventory': '75500', 'Product Name': '27in FHD Monitor', 'Revenue': '52.4965'}}], 'table_id': 'file_0_view_0'}]}
import gurobipy
from gurobipy import GRB

def solve_problem():
    tables = CSVQA_DATA.get('tables', [])
    target_table = None
    for t in tables:
        if t.get('table_id') == 'file_0_view_0':
            target_table = t
            break
    if target_table is None:
        raise ValueError("Required table 'file_0_view_0' not found in CSVQA_DATA")
    records = target_table.get('records', [])
    if not records:
        raise ValueError('No records found in the target table')
    revenue = {}
    demand = {}
    initial = {}
    for rec in records:
        vals = rec.get('values', {})
        for col in ('Product Name', 'Revenue', 'Demand', 'Initial Inventory'):
            if col not in vals:
                raise ValueError(f"Missing column '{col}' in record {rec.get('data_row_number')}")
        prod = vals['Product Name']
        try:
            rev = float(vals['Revenue'])
            dem = int(vals['Demand'])
            init = int(vals['Initial Inventory'])
        except ValueError as e:
            raise ValueError(f"Invalid numeric value in record {rec.get('data_row_number')}: {e}")
        if prod in revenue:
            raise ValueError(f"Duplicate product name '{prod}' found in records")
        revenue[prod] = rev
        demand[prod] = dem
        initial[prod] = init
    I = list(revenue.keys())
    m = gurobipy.Model()
    m.setParam('MIPGap', 1e-06)
    x = m.addVars(I, lb=0, vtype=GRB.INTEGER, name='')
    for i in I:
        m.addConstr(x[i] <= demand[i], name=f'demand_{i}')
        m.addConstr(x[i] <= initial[i], name=f'initial_{i}')
    m.setObjective(gurobipy.quicksum((revenue[i] * x[i] for i in I)), GRB.MAXIMIZE)
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()
