CSVQA_DATA = {'query': 'The supermarket offers various products with revenue data provided in the ‘Revenue’ column. The retailer aims to maximize total revenue by focusing on products classified under ‘Fashion’. Inventory levels are given in the ‘Initial Inventory’ column. Demand quantities are provided in the ‘Demand’ column and are assumed to be deterministic and known in advance. Decision variables x_i represent the number of units of each ‘Fashion’ product i that will be fulfilled.', 'relationships': [], 'tables': [{'columns': ['Product Name', 'Revenue', 'Demand', 'Initial Inventory'], 'file_name': 'SupermarketSales.csv', 'records': [{'data_row_number': 168, 'values': {'Demand': '12', 'Initial Inventory': '80', 'Product Name': 'Fashion accessories_10.18', 'Revenue': '10.18'}}, {'data_row_number': 169, 'values': {'Demand': '2', 'Initial Inventory': '10', 'Product Name': 'Fashion accessories_12.09', 'Revenue': '12.09'}}, {'data_row_number': 170, 'values': {'Demand': '10', 'Initial Inventory': '80', 'Product Name': 'Fashion accessories_12.19', 'Revenue': '12.19'}}, {'data_row_number': 171, 'values': {'Demand': '2', 'Initial Inventory': '10', 'Product Name': 'Fashion accessories_12.54', 'Revenue': '12.54'}}, {'data_row_number': 172, 'values': {'Demand': '2', 'Initial Inventory': '10', 'Product Name': 'Fashion accessories_12.78', 'Revenue': '12.78'}}, {'data_row_number': 173, 'values': {'Demand': '5', 'Initial Inventory': '40', 'Product Name': 'Fashion accessories_14.48', 'Revenue': '14.48'}}, {'data_row_number': 174, 'values': {'Demand': '2', 'Initial Inventory': '10', 'Product Name': 'Fashion accessories_15.43', 'Revenue': '15.43'}}, {'data_row_number': 175, 'values': {'Demand': '2', 'Initial Inventory': '10', 'Product Name': 'Fashion accessories_15.5', 'Revenue': '15.5'}}, {'data_row_number': 176, 'values': {'Demand': '11', 'Initial Inventory': '80', 'Product Name': 'Fashion accessories_15.62', 'Revenue': '15.62'}}, {'data_row_number': 177, 'values': {'Demand': '2', 'Initial Inventory': '10', 'Product Name': 'Fashion accessories_16.28', 'Revenue': '16.28'}}, {'data_row_number': 178, 'values': {'Demand': '6', 'Initial Inventory': '40', 'Product Name': 'Fashion accessories_16.45', 'Revenue': '16.45'}}, {'data_row_number': 179, 'values': {'Demand': '9', 'Initial Inventory': '60', 'Product Name': 'Fashion accessories_17.48', 'Revenue': '17.48'}}, {'data_row_number': 180, 'values': {'Demand': '14', 'Initial Inventory': '100', 'Product Name': 'Fashion accessories_17.49', 'Revenue': '17.49'}}, {'data_row_number': 181, 'values': {'Demand': '6', 'Initial Inventory': '40', 'Product Name': 'Fashion accessories_17.87', 'Revenue': '17.87'}}, {'data_row_number': 182, 'values': {'Demand': '8', 'Initial Inventory': '50', 'Product Name': 'Fashion accessories_17.94', 'Revenue': '17.94'}}, {'data_row_number': 183, 'values': {'Demand': '6', 'Initial Inventory': '40', 'Product Name': 'Fashion accessories_18.08', 'Revenue': '18.08'}}, {'data_row_number': 184, 'values': {'Demand': '14', 'Initial Inventory': '100', 'Product Name': 'Fashion accessories_19.66', 'Revenue': '19.66'}}, {'data_row_number': 185, 'values': {'Demand': '2', 'Initial Inventory': '10', 'Product Name': 'Fashion accessories_19.7', 'Revenue': '19.7'}}, {'data_row_number': 186, 'values': {'Demand': '14', 'Initial Inventory': '100', 'Product Name': 'Fashion accessories_19.77', 'Revenue': '19.77'}}, {'data_row_number': 187, 'values': {'Demand': '11', 'Initial Inventory': '90', 'Product Name': 'Fashion accessories_20.01', 'Revenue': '20.01'}}, {'data_row_number': 188, 'values': {'Demand': '2', 'Initial Inventory': '10', 'Product Name': 'Fashion accessories_21.32', 'Revenue': '21.32'}}, {'data_row_number': 189, 'values': {'Demand': '3', 'Initial Inventory': '20', 'Product Name': 'Fashion accessories_21.48', 'Revenue': '21.48'}}, {'data_row_number': 190, 'values': {'Demand': '8', 'Initial Inventory': '50', 'Product Name': 'Fashion accessories_21.94', 'Revenue': '21.94'}}, {'data_row_number': 191, 'values': {'Demand': '10', 'Initial Inventory': '80', 'Product Name': 'Fashion accessories_22.32', 'Revenue': '22.32'}}, {'data_row_number': 192, 'values': {'Demand': '10', 'Initial Inventory': '70', 'Product Name': 'Fashion accessories_22.51', 'Revenue': '22.51'}}, {'data_row_number': 193, 'values': {'Demand': '7', 'Initial Inventory': '50', 'Product Name': 'Fashion accessories_23.82', 'Revenue': '23.82'}}, {'data_row_number': 194, 'values': {'Demand': '11', 'Initial Inventory': '80', 'Product Name': 'Fashion accessories_25.42', 'Revenue': '25.42'}}, {'data_row_number': 195, 'values': {'Demand': '11', 'Initial Inventory': '70', 'Product Name': 'Fashion accessories_25.56', 'Revenue': '25.56'}}, {'data_row_number': 196, 'values': {'Demand': '5', 'Initial Inventory': '30', 'Product Name': 'Fashion accessories_27.02', 'Revenue': '27.02'}}, {'data_row_number': 197, 'values': {'Demand': '3', 'Initial Inventory': '20', 'Product Name': 'Fashion accessories_27.18', 'Revenue': '27.18'}}, {'data_row_number': 198, 'values': {'Demand': '8', 'Initial Inventory': '60', 'Product Name': 'Fashion accessories_27.38', 'Revenue': '27.38'}}, {'data_row_number': 199, 'values': {'Demand': '13', 'Initial Inventory': '100', 'Product Name': 'Fashion accessories_29.42', 'Revenue': '29.42'}}, {'data_row_number': 200, 'values': {'Demand': '7', 'Initial Inventory': '50', 'Product Name': 'Fashion accessories_29.56', 'Revenue': '29.56'}}, {'data_row_number': 201, 'values': {'Demand': '14', 'Initial Inventory': '100', 'Product Name': 'Fashion accessories_30.14', 'Revenue': '30.14'}}, {'data_row_number': 202, 'values': {'Demand': '4', 'Initial Inventory': '30', 'Product Name': 'Fashion accessories_30.37', 'Revenue': '30.37'}}, {'data_row_number': 203, 'values': {'Demand': '2', 'Initial Inventory': '10', 'Product Name': 'Fashion accessories_30.61', 'Revenue': '30.61'}}, {'data_row_number': 204, 'values': {'Demand': '2', 'Initial Inventory': '10', 'Product Name': 'Fashion accessories_30.62', 'Revenue': '30.62'}}, {'data_row_number': 205, 'values': {'Demand': '14', 'Initial Inventory': '90', 'Product Name': 'Fashion accessories_31.73', 'Revenue': '31.73'}}, {'data_row_number': 206, 'values': {'Demand': '2', 'Initial Inventory': '10', 'Product Name': 'Fashion accessories_31.9', 'Revenue': '31.9'}}, {'data_row_number': 207, 'values': {'Demand': '6', 'Initial Inventory': '40', 'Product Name': 'Fashion accessories_32.62', 'Revenue': '32.62'}}, {'data_row_number': 208, 'values': {'Demand': '2', 'Initial Inventory': '10', 'Product Name': 'Fashion accessories_33.52', 'Revenue': '33.52'}}, {'data_row_number': 209, 'values': {'Demand': '2', 'Initial Inventory': '10', 'Product Name': 'Fashion accessories_33.63', 'Revenue': '33.63'}}, {'data_row_number': 210, 'values': {'Demand': '3', 'Initial Inventory': '20', 'Product Name': 'Fashion accessories_34.7', 'Revenue': '34.7'}}, {'data_row_number': 211, 'values': {'Demand': '14', 'Initial Inventory': '100', 'Product Name': 'Fashion accessories_35.19', 'Revenue': '35.19'}}, {'data_row_number': 212, 'values': {'Demand': '12', 'Initial Inventory': '90', 'Product Name': 'Fashion accessories_36.51', 'Revenue': '36.51'}}, {'data_row_number': 213, 'values': {'Demand': '7', 'Initial Inventory': '50', 'Product Name': 'Fashion accessories_36.85', 'Revenue': '36.85'}}, {'data_row_number': 214, 'values': {'Demand': '6', 'Initial Inventory': '40', 'Product Name': 'Fashion accessories_37.15', 'Revenue': '37.15'}}, {'data_row_number': 215, 'values': {'Demand': '13', 'Initial Inventory': '100', 'Product Name': 'Fashion accessories_37.55', 'Revenue': '37.55'}}, {'data_row_number': 216, 'values': {'Demand': '14', 'Initial Inventory': '100', 'Product Name': 'Fashion accessories_37.95', 'Revenue': '37.95'}}], 'table_id': 'file_0_view_0'}]}
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
    product_names = []
    revenue = {}
    demand = {}
    inventory = {}
    for rec in records:
        values = rec.get('values', {})
        pname = values.get('Product Name')
        rev_str = values.get('Revenue')
        dem_str = values.get('Demand')
        inv_str = values.get('Initial Inventory')
        if pname is None or rev_str is None or dem_str is None or (inv_str is None):
            raise ValueError(f"Missing required field in record {rec.get('data_row_number')}.")
        try:
            rev_val = float(rev_str)
            dem_val = int(dem_str)
            inv_val = int(inv_str)
        except ValueError as e:
            raise ValueError(f"Invalid numeric conversion in record {rec.get('data_row_number')}: {e}")
        product_names.append(pname)
        revenue[pname] = rev_val
        demand[pname] = dem_val
        inventory[pname] = inv_val
    for pname in product_names:
        if pname not in revenue or pname not in demand or pname not in inventory:
            raise ValueError(f'Missing parameter for product {pname}.')
    m = gurobipy.Model()
    m.setParam('MIPGap', 1e-06)
    x = m.addVars(product_names, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gurobipy.quicksum((revenue[p] * x[p] for p in product_names)), GRB.MAXIMIZE)
    for p in product_names:
        m.addConstr(x[p] <= demand[p], name=f'demand_{p}')
    for p in product_names:
        m.addConstr(x[p] <= inventory[p], name=f'inventory_{p}')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()
