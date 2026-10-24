CSVQA_DATA = {'query': 'The department store is hosting a promotional event featuring various top-selling items. Revenue data is available in the ‘Revenue’ column. The retailer aims to maximize total revenue using the initial inventory of products classified under ‘27in’. Inventory levels are detailed in the ‘Initial Inventory’ column. Demand quantities are provided in the ‘Demand’ column and are assumed to be deterministic and known in advance. Decision variables x_i indicate the number of units of each ‘27in’ product i that will be fulfilled.', 'relationships': [], 'tables': [{'columns': ['Product Name', 'Revenue', 'Demand', 'Initial Inventory'], 'file_name': 'Salesorders.csv', 'records': [{'data_row_number': 1, 'values': {'Demand': '12474', 'Initial Inventory': '62440', 'Product Name': '27in 4K Gaming Monitor', 'Revenue': '261.2933'}}, {'data_row_number': 2, 'values': {'Demand': '15057', 'Initial Inventory': '75500', 'Product Name': '27in FHD Monitor', 'Revenue': '52.4965'}}], 'table_id': 'file_0_view_0'}]}
import gurobipy
from gurobipy import GRB

def solve_problem():
    table = CSVQA_DATA['tables'][0]
    records = table['records']
    I = []
    rev = {}
    dem = {}
    inv = {}
    for rec in records:
        values = rec['values']
        try:
            prod_name = values['Product Name']
            rev_val = float(values['Revenue'])
            dem_val = float(values['Demand'])
            inv_val = float(values['Initial Inventory'])
        except KeyError as e:
            raise ValueError(f"Missing required field {e} in record {rec['data_row_number']}")
        except ValueError as e:
            raise ValueError(f"Invalid numeric value in record {rec['data_row_number']}: {e}")
        I.append(prod_name)
        rev[prod_name] = rev_val
        dem[prod_name] = dem_val
        inv[prod_name] = inv_val
    for i in I:
        if i not in rev or i not in dem or i not in inv:
            raise ValueError(f'Missing parameters for product {i}')
    model = gurobipy.Model('27in_product_sales')
    model.setParam('MIPGap', 1e-06)
    x = model.addVars(I, lb=0, vtype=GRB.CONTINUOUS, name='')
    model.addConstrs((x[i] <= dem[i] for i in I), name='')
    model.addConstrs((x[i] <= inv[i] for i in I), name='')
    model.setObjective(gurobipy.quicksum((rev[i] * x[i] for i in I)), GRB.MAXIMIZE)
    model.optimize()
    if model.Status == GRB.OPTIMAL:
        print(f'ObjVal: {model.ObjVal}')
        for var in model.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {model.Status}')
    return model
m = solve_problem()
