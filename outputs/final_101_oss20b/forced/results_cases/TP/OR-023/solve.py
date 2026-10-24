CSVQA_DATA = {'query': 'The supermarket offers a variety of top-selling products with revenue data in the ‘Revenue’ column. The retailer aims to maximize total revenue using the initial inventory of products classified under ‘ELE-S’. Inventory levels are given in the ‘Initial Inventory’ column. Demand quantities are specified in the ‘Demand’ column and are assumed to be deterministic and known in advance. Decision variables x_i indicate the number of units of each ‘ELE-S’ product i that the company plans to fulfill.', 'relationships': [], 'tables': [{'columns': ['Product_Reference', 'Revenue', 'Demand', 'Initial Inventory'], 'file_name': 'SalesStoreoverview.csv', 'records': [{'data_row_number': 36, 'values': {'Demand': '295', 'Initial Inventory': '2000.0', 'Product_Reference': 'ELE-SMA-10000463', 'Revenue': '4.0'}}, {'data_row_number': 37, 'values': {'Demand': '1002', 'Initial Inventory': '7000.0', 'Product_Reference': 'ELE-SMA-10000487', 'Revenue': '14.0'}}, {'data_row_number': 38, 'values': {'Demand': '958', 'Initial Inventory': '7000.0', 'Product_Reference': 'ELE-SMA-10003333', 'Revenue': '14.0'}}, {'data_row_number': 39, 'values': {'Demand': '777', 'Initial Inventory': '6000.0', 'Product_Reference': 'ELE-SMA-10009012', 'Revenue': '4.0'}}, {'data_row_number': 40, 'values': {'Demand': '271', 'Initial Inventory': '2000.0', 'Product_Reference': 'ELE-SMA-10009999', 'Revenue': '4.0'}}, {'data_row_number': 41, 'values': {'Demand': '244', 'Initial Inventory': '2000.0', 'Product_Reference': 'ELE-SMA-10011234', 'Revenue': '4.0'}}, {'data_row_number': 42, 'values': {'Demand': '990', 'Initial Inventory': '7000.0', 'Product_Reference': 'ELE-SMA-10027456', 'Revenue': '14.0'}}, {'data_row_number': 43, 'values': {'Demand': '1000', 'Initial Inventory': '7000.0', 'Product_Reference': 'ELE-SMA-10028567', 'Revenue': '14.0'}}, {'data_row_number': 44, 'values': {'Demand': '169', 'Initial Inventory': '1200.0', 'Product_Reference': 'ELE-SPE-10000484', 'Revenue': '2.4'}}, {'data_row_number': 45, 'values': {'Demand': '155', 'Initial Inventory': '1200.0', 'Product_Reference': 'ELE-SPE-10003030', 'Revenue': '2.4'}}, {'data_row_number': 46, 'values': {'Demand': '327', 'Initial Inventory': '2400.0', 'Product_Reference': 'ELE-SPE-10024123', 'Revenue': '2.4'}}, {'data_row_number': 47, 'values': {'Demand': '174', 'Initial Inventory': '1200.0', 'Product_Reference': 'ELE-SPE-10025234', 'Revenue': '2.4'}}], 'table_id': 'file_0_view_0'}]}
import gurobipy
from gurobipy import GRB

def solve_problem():
    tables = CSVQA_DATA.get('tables', [])
    target_table = None
    for tbl in tables:
        if tbl.get('table_id') == 'file_0_view_0':
            target_table = tbl
            break
    if target_table is None:
        raise ValueError("Required table 'file_0_view_0' not found in CSVQA_DATA")
    records = target_table.get('records', [])
    I = []
    r = {}
    d = {}
    s = {}
    for rec in records:
        vals = rec.get('values', {})
        prod_ref = vals.get('Product_Reference')
        if prod_ref is None:
            raise ValueError(f"Missing Product_Reference in record {rec.get('data_row_number')}")
        if 'Revenue' not in vals or 'Demand' not in vals or 'Initial Inventory' not in vals:
            raise ValueError(f"Missing parameter in record {rec.get('data_row_number')}")
        I.append(prod_ref)
        r[prod_ref] = float(vals['Revenue'])
        d[prod_ref] = int(vals['Demand'])
        s[prod_ref] = float(vals['Initial Inventory'])
    for i in I:
        if i not in r or i not in d or i not in s:
            raise ValueError(f'Missing parameters for product {i}')
    m = gurobipy.Model('EleS_Revenue_Maximization')
    m.setParam('MIPGap', 1e-06)
    x = m.addVars(I, lb=0, vtype=GRB.CONTINUOUS, name='')
    m.setObjective(gurobipy.quicksum((r[i] * x[i] for i in I)), GRB.MAXIMIZE)
    for i in I:
        m.addConstr(x[i] <= d[i], name=f'demand_{i}')
        m.addConstr(x[i] <= s[i], name=f'supply_{i}')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()
