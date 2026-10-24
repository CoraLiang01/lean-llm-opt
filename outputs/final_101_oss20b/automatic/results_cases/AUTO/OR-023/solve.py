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
        raise ValueError("Required table 'file_0_view_0' not found in CSVQA_DATA.")
    records = target_table.get('records', [])
    if not records:
        raise ValueError('No records found in the target table.')
    R = {}
    D = {}
    I = {}
    I_set = []
    for rec in records:
        vals = rec.get('values', {})
        prod_ref = vals.get('Product_Reference')
        if prod_ref is None:
            raise ValueError(f"Missing Product_Reference in record {rec.get('data_row_number')}.")
        if not prod_ref.startswith('ELE-S'):
            continue
        revenue_str = vals.get('Revenue')
        demand_str = vals.get('Demand')
        inventory_str = vals.get('Initial Inventory')
        if revenue_str is None or demand_str is None or inventory_str is None:
            raise ValueError(f"Missing parameter in record {rec.get('data_row_number')} for product {prod_ref}.")
        try:
            revenue = float(revenue_str)
            demand = int(demand_str)
            inventory = float(inventory_str)
        except ValueError as e:
            raise ValueError(f"Invalid numeric value in record {rec.get('data_row_number')} for product {prod_ref}: {e}")
        R[prod_ref] = revenue
        D[prod_ref] = demand
        I[prod_ref] = inventory
        I_set.append(prod_ref)
    if not I_set:
        raise ValueError("No products with prefix 'ELE-S' found after filtering.")
    model = gurobipy.Model('EleS_Revenue_Max')
    model.setParam('MIPGap', 1e-06)
    x = model.addVars(I_set, vtype=GRB.INTEGER, lb=0, name='')
    model.setObjective(gurobipy.quicksum((R[i] * x[i] for i in I_set)), GRB.MAXIMIZE)
    for i in I_set:
        model.addConstr(x[i] <= D[i], name=f'demand_{i}')
        model.addConstr(x[i] <= I[i], name=f'inventory_{i}')
    model.optimize()
    return model
m = solve_problem()
