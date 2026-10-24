CSVQA_DATA = {'query': 'The supermarket offers a variety of top-selling products, with associated data provided in the ‘Revenue’ column. Each product has its own demand level. The retailer aims to maximize total revenue by focusing on the initial inventory of products classified under ‘Baby’. Inventory levels are detailed in the ‘Initial Inventory’ column. During the sales horizon, no restocking is allowed. Demand quantities for ‘Baby’ products are specified in the ‘Demand’ column and are assumed to be deterministic and known in advance. The decision variables x_i represent the number of units of each ‘Baby’ product i that the retailer plans to fulfill.', 'relationships': [], 'tables': [{'columns': ['Product Name', 'Revenue', 'Demand', 'Initial Inventory'], 'file_name': 'Salesdata.csv', 'records': [{'data_row_number': 0, 'values': {'Demand': '3066513', 'Initial Inventory': '22749210', 'Product Name': 'Baby Food_255.28', 'Revenue': '255.28'}}], 'table_id': 'file_0_view_0'}]}
import gurobipy
from gurobipy import GRB

def solve_problem():
    tables = CSVQA_DATA.get('tables', [])
    if not tables:
        raise ValueError('No tables found in CSVQA_DATA')
    table = None
    for t in tables:
        if t.get('table_id') == 'file_0_view_0':
            table = t
            break
    if table is None:
        raise ValueError("Required table 'file_0_view_0' not found")
    records = table.get('records', [])
    revenue_dict = {}
    demand_dict = {}
    inv_dict = {}
    index_set = []
    for rec in records:
        data_row_number = rec.get('data_row_number')
        if data_row_number is None:
            raise ValueError('Record missing data_row_number')
        values = rec.get('values', {})
        try:
            prod_name = values['Product Name']
            rev_str = values['Revenue']
            dem_str = values['Demand']
            inv_str = values['Initial Inventory']
        except KeyError as e:
            raise ValueError(f'Missing column in record {data_row_number}: {e}')
        if not prod_name.startswith('Baby'):
            continue
        try:
            rev = float(rev_str)
            dem = int(dem_str)
            inv = int(inv_str)
        except ValueError as e:
            raise ValueError(f'Invalid numeric value in record {data_row_number}: {e}')
        revenue_dict[data_row_number] = rev
        demand_dict[data_row_number] = dem
        inv_dict[data_row_number] = inv
        index_set.append(data_row_number)
    if not index_set:
        model = gurobipy.Model('BabyProductRevenueMax')
        model.setParam('MIPGap', 1e-06)
        return model
    model = gurobipy.Model('BabyProductRevenueMax')
    model.setParam('MIPGap', 1e-06)
    ub_dict = {i: min(demand_dict[i], inv_dict[i]) for i in index_set}
    x = model.addVars(index_set, lb=0, ub=ub_dict, vtype=GRB.INTEGER, name='')
    model.setObjective(gurobipy.quicksum((revenue_dict[i] * x[i] for i in index_set)), GRB.MAXIMIZE)
    for i in index_set:
        model.addConstr(x[i] <= demand_dict[i], name=f'demand_{i}')
        model.addConstr(x[i] <= inv_dict[i], name=f'inv_{i}')
    model.optimize()
    if model.Status == GRB.OPTIMAL:
        print(f'ObjVal: {model.ObjVal}')
        for var in model.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {model.Status}')
    return model
m = solve_problem()
