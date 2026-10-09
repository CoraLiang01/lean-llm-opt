CSVQA_DATA = {'route': 'NRM',
 'tables': [{'table_id': 'file_0_view_0',
             'file_index': 0,
             'file_name': 'Salesdata.csv',
             'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM9/Salesdata.csv',
             'role': 'file_0',
             'columns': ['Product Name', 'Revenue', 'Demand', 'Initial Inventory'],
             'original_rows': 12,
             'returned_rows': 12,
             'filters': {'logic': 'and', 'conditions': []},
             'records': [{'source_row': 0,
                          'values': {'Product Name': 'Baby Food_255.28',
                                     'Revenue': '255.28',
                                     'Demand': '3066513',
                                     'Initial Inventory': '22749210'}},
                         {'source_row': 1,
                          'values': {'Product Name': 'Beverages_47.45',
                                     'Revenue': '47.45',
                                     'Demand': '2961484',
                                     'Initial Inventory': '22049510'}},
                         {'source_row': 2,
                          'values': {'Product Name': 'Cereal_205.7',
                                     'Revenue': '205.7',
                                     'Demand': '2621950',
                                     'Initial Inventory': '19459680'}},
                         {'source_row': 3,
                          'values': {'Product Name': 'Clothes_109.28',
                                     'Revenue': '109.28',
                                     'Demand': '2660974',
                                     'Initial Inventory': '19754410'}},
                         {'source_row': 4,
                          'values': {'Product Name': 'Cosmetics_437.2',
                                     'Revenue': '437.2',
                                     'Demand': '2896197',
                                     'Initial Inventory': '21366410'}},
                         {'source_row': 5,
                          'values': {'Product Name': 'Fruits_9.33',
                                     'Revenue': '9.33',
                                     'Demand': '3169426',
                                     'Initial Inventory': '23410830'}},
                         {'source_row': 6,
                          'values': {'Product Name': 'Household_668.27',
                                     'Revenue': '668.27',
                                     'Demand': '2846953',
                                     'Initial Inventory': '20986130'}},
                         {'source_row': 7,
                          'values': {'Product Name': 'Meat_421.89',
                                     'Revenue': '421.89',
                                     'Demand': '2546972',
                                     'Initial Inventory': '19011970'}},
                         {'source_row': 8,
                          'values': {'Product Name': 'Office Supplies_651.21',
                                     'Revenue': '651.21',
                                     'Demand': '2855686',
                                     'Initial Inventory': '21062780'}},
                         {'source_row': 9,
                          'values': {'Product Name': 'Personal Care_81.73',
                                     'Revenue': '81.73',
                                     'Demand': '2855360',
                                     'Initial Inventory': '21265920'}},
                         {'source_row': 10,
                          'values': {'Product Name': 'Snacks_152.58',
                                     'Revenue': '152.58',
                                     'Demand': '2592261',
                                     'Initial Inventory': '19155280'}},
                         {'source_row': 11,
                          'values': {'Product Name': 'Vegetables_154.06',
                                     'Revenue': '154.06',
                                     'Demand': '2826603',
                                     'Initial Inventory': '20867490'}}]}],
 'relationships': [],
 'ignored_file_indices': [],
 'validation': {'status': 'PYTHON_FULL_CSV'}}
import gurobipy as gp
from gurobipy import GRB
import re
table_id = 'file_0_view_0'
if 'CSVQA_DATA' not in globals():
    raise RuntimeError('CSVQA_DATA is not defined at execution.')
tables = CSVQA_DATA.get('tables', [])
table = next((t for t in tables if t.get('table_id') == table_id), None)
if table is None:
    raise RuntimeError(f'Table {table_id} not found in CSVQA_DATA.')
records = table.get('records', [])
if not records:
    raise RuntimeError(f'No records found in table {table_id}.')
baby_items = []
revenue = {}
demand = {}
inventory = {}
for rec in records:
    vals = rec.get('values', {})
    pname = vals.get('Product Name', '')
    if not isinstance(pname, str):
        continue
    if re.search('Baby', pname, re.IGNORECASE):
        baby_items.append(pname)
        try:
            rev = float(vals['Revenue'])
            dem = float(vals['Demand'])
            inv = float(vals['Initial Inventory'])
        except Exception as e:
            raise ValueError(f"Non-numeric parameter for product '{pname}': {e}")
        revenue[pname] = rev
        demand[pname] = dem
        inventory[pname] = inv
if not baby_items:
    raise ValueError("No products classified under 'Baby' found in the data.")
for pname in baby_items:
    if pname not in revenue or pname not in demand or pname not in inventory:
        raise ValueError(f"Missing parameter(s) for product '{pname}'.")
m = gp.Model('Baby_Product_Revenue_Max')
x_vars = m.addVars(baby_items, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((revenue[i] * x_vars[i] for i in baby_items)), GRB.MAXIMIZE)
for i in baby_items:
    upper = min(inventory[i], demand[i])
    m.addConstr(x_vars[i] <= upper, name=f'ub_{i}')
    m.addConstr(x_vars[i] >= 0, name=f'lb_{i}')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in x_vars.values():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')