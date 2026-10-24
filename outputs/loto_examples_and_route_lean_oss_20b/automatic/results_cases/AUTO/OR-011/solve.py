CSVQA_DATA = {'query': 'The supermarket offers a variety of top-selling products, with associated revenue data provided in the ‘Revenue’ column. Each product has its own demand level during the sales horizon. The company’s objective is to maximize total revenue by allocating the available inventory of products classified under ‘id999’. The initial inventory levels for the ‘id999’ products are detailed in the ‘Initial Inventory’ column. During the sales horizon, no restocking is allowed, and there are no in-transit inventories.\n\nDemand for each product during the sales period is assumed to be deterministic and known in advance, with demand quantities specified in the ‘Demand’ column. The decision variables x_i represent the number of units of each ‘id999’ product i that the company plans to fulfill, where each x_i is a non-negative integer. Because fulfilled orders cannot exceed either the available inventory or the realized demand, the fulfillment quantities must satisfy both inventory and demand constraints.', 'relationships': [], 'tables': [{'columns': ['id_number', 'Revenue', 'Demand', 'Initial Inventory'], 'file_name': 'OnlineRetailSalesDataset.csv', 'records': [{'data_row_number': 899, 'values': {'Demand': '8171', 'Initial Inventory': '56450', 'Revenue': '434.74', 'id_number': 'id999'}}], 'table_id': 'file_0_view_0'}]}
import gurobipy
from gurobipy import GRB

def solve_problem():
    table = None
    for tbl in CSVQA_DATA['tables']:
        if tbl.get('table_id') == 'file_0_view_0':
            table = tbl
            break
    if table is None:
        raise ValueError("Required table 'file_0_view_0' not found in CSVQA_DATA")
    I = []
    Rev = {}
    Dem = {}
    Inv = {}
    for record in table.get('records', []):
        values = record.get('values', {})
        id_num = values.get('id_number', '').strip()
        if not id_num.startswith('id999'):
            continue
        if 'Revenue' not in values or 'Demand' not in values or 'Initial Inventory' not in values:
            raise ValueError(f"Missing required columns for record {record.get('data_row_number')}")
        try:
            rev_val = float(values['Revenue'])
            dem_val = int(values['Demand'])
            inv_val = int(values['Initial Inventory'])
        except ValueError as e:
            raise ValueError(f"Invalid numeric value in record {record.get('data_row_number')}: {e}")
        I.append(id_num)
        Rev[id_num] = rev_val
        Dem[id_num] = dem_val
        Inv[id_num] = inv_val
    if not I:
        raise ValueError("No products with id_number starting with 'id999' found")
    model = gurobipy.Model()
    model.setParam('MIPGap', 1e-06)
    x = model.addVars(I, lb=0, vtype=GRB.INTEGER, name='')
    model.setObjective(gurobipy.quicksum((Rev[i] * x[i] for i in I)), GRB.MAXIMIZE)
    for i in I:
        model.addConstr(x[i] <= Inv[i], name=f'inv_{i}')
        model.addConstr(x[i] <= Dem[i], name=f'dem_{i}')
    model.optimize()
    return model
m = solve_problem()
