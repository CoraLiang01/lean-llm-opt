CSVQA_DATA = {'route': 'NRM',
 'tables': [{'table_id': 'file_0_view_0',
             'file_index': 0,
             'file_name': 'EuropeSalesRecords.csv',
             'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM24/EuropeSalesRecords.csv',
             'role': 'file_0',
             'columns': ['Product Name', 'Revenue', 'Demand', 'Initial Inventory'],
             'original_rows': 12,
             'returned_rows': 12,
             'filters': {'logic': 'and', 'conditions': []},
             'records': [{'source_row': 0,
                          'values': {'Product Name': 'Baby Food_255.28',
                                     'Revenue': '255.28',
                                     'Demand': '765850',
                                     'Initial Inventory': '5627060'}},
                         {'source_row': 1,
                          'values': {'Product Name': 'Beverages_47.45',
                                     'Revenue': '47.45',
                                     'Demand': '825453',
                                     'Initial Inventory': '6131330'}},
                         {'source_row': 2,
                          'values': {'Product Name': 'Cereal_205.7',
                                     'Revenue': '205.7',
                                     'Demand': '627481',
                                     'Initial Inventory': '4656850'}},
                         {'source_row': 3,
                          'values': {'Product Name': 'Clothes_109.28',
                                     'Revenue': '109.28',
                                     'Demand': '800987',
                                     'Initial Inventory': '5913850'}},
                         {'source_row': 4,
                          'values': {'Product Name': 'Cosmetics_437.2',
                                     'Revenue': '437.2',
                                     'Demand': '718806',
                                     'Initial Inventory': '5332910'}},
                         {'source_row': 5,
                          'values': {'Product Name': 'Fruits_9.33',
                                     'Revenue': '9.33',
                                     'Demand': '798999',
                                     'Initial Inventory': '5916720'}},
                         {'source_row': 6,
                          'values': {'Product Name': 'Household_668.27',
                                     'Revenue': '668.27',
                                     'Demand': '591313',
                                     'Initial Inventory': '4402490'}},
                         {'source_row': 7,
                          'values': {'Product Name': 'Meat_421.89',
                                     'Revenue': '421.89',
                                     'Demand': '713606',
                                     'Initial Inventory': '5333760'}},
                         {'source_row': 8,
                          'values': {'Product Name': 'Office Supplies_651.21',
                                     'Revenue': '651.21',
                                     'Demand': '838862',
                                     'Initial Inventory': '6176410'}},
                         {'source_row': 9,
                          'values': {'Product Name': 'Personal Care_81.73',
                                     'Revenue': '81.73',
                                     'Demand': '756433',
                                     'Initial Inventory': '5604800'}},
                         {'source_row': 10,
                          'values': {'Product Name': 'Snacks_152.58',
                                     'Revenue': '152.58',
                                     'Demand': '655310',
                                     'Initial Inventory': '4901600'}},
                         {'source_row': 11,
                          'values': {'Product Name': 'Vegetables_154.06',
                                     'Revenue': '154.06',
                                     'Demand': '786187',
                                     'Initial Inventory': '5825440'}}]}],
 'relationships': [],
 'ignored_file_indices': [],
 'validation': {'status': 'PYTHON_FULL_CSV'}}
import re
import gurobipy as gp
from gurobipy import GRB
if 'CSVQA_DATA' not in globals():
    raise RuntimeError('CSVQA_DATA not found in global scope.')
tables = CSVQA_DATA.get('tables', [])
table = None
for t in tables:
    if t.get('table_id') == 'file_0_view_0':
        table = t
        break
if table is None:
    raise RuntimeError('Table file_0_view_0 not found in CSVQA_DATA.')
records = table.get('records', [])
if not records:
    raise RuntimeError('No records found in file_0_view_0.')
baby_items = []
revenue = {}
demand = {}
inventory = {}
for rec in records:
    vals = rec.get('values', {})
    pname = vals.get('Product Name', '')
    if not re.search('\\bBaby\\b', pname, re.IGNORECASE):
        continue
    baby_items.append(pname)
    try:
        revenue[pname] = float(vals['Revenue'])
    except Exception:
        raise ValueError(f"Invalid Revenue for {pname}: {vals.get('Revenue')}")
    try:
        demand[pname] = int(float(vals['Demand']))
    except Exception:
        raise ValueError(f"Invalid Demand for {pname}: {vals.get('Demand')}")
    try:
        inventory[pname] = int(float(vals['Initial Inventory']))
    except Exception:
        raise ValueError(f"Invalid Initial Inventory for {pname}: {vals.get('Initial Inventory')}")
if not baby_items:
    raise RuntimeError("No 'Baby' products found in file_0_view_0.")
for i in baby_items:
    if i not in revenue or i not in demand or i not in inventory:
        raise RuntimeError(f'Missing data for product {i}.')
m = gp.Model('Baby_Product_Revenue_Max')
x_vars = m.addVars(baby_items, lb=0, ub=[min(demand[i], inventory[i]) for i in baby_items], vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((revenue[i] * x_vars[i] for i in baby_items)), GRB.MAXIMIZE)
m.addConstrs((x_vars[i] <= demand[i] for i in baby_items), name='')
m.addConstrs((x_vars[i] <= inventory[i] for i in baby_items), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')