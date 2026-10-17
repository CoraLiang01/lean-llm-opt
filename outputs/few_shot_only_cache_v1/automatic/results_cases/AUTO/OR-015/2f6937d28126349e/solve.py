CSVQA_DATA = {'tables': [{'table_id': 'file_0_view_0',
             'file_index': 0,
             'file_name': 'RestaurantSalesreport.csv',
             'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM6/RestaurantSalesreport.csv',
             'columns': ['Product Name', 'Revenue', 'Demand', 'Initial Inventory'],
             'original_rows': 7,
             'returned_rows': 7,
             'records': [{'source_row': 0,
                          'values': {'Product Name': 'Aalopuri',
                                     'Revenue': '20',
                                     'Demand': '1483',
                                     'Initial Inventory': '10440.0'}},
                         {'source_row': 1,
                          'values': {'Product Name': 'Cold coffee',
                                     'Revenue': '40',
                                     'Demand': '1918',
                                     'Initial Inventory': '13610.0'}},
                         {'source_row': 2,
                          'values': {'Product Name': 'Frankie',
                                     'Revenue': '50',
                                     'Demand': '1623',
                                     'Initial Inventory': '11500.0'}},
                         {'source_row': 3,
                          'values': {'Product Name': 'Panipuri',
                                     'Revenue': '20',
                                     'Demand': '1720',
                                     'Initial Inventory': '12260.0'}},
                         {'source_row': 4,
                          'values': {'Product Name': 'Sandwich',
                                     'Revenue': '60',
                                     'Demand': '1558',
                                     'Initial Inventory': '10970.0'}},
                         {'source_row': 5,
                          'values': {'Product Name': 'Sugarcane juice',
                                     'Revenue': '25',
                                     'Demand': '1791',
                                     'Initial Inventory': '12780.0'}},
                         {'source_row': 6,
                          'values': {'Product Name': 'Vadapav',
                                     'Revenue': '20',
                                     'Demand': '1426',
                                     'Initial Inventory': '10060.0'}}]}]}
import gurobipy as gp
from gurobipy import GRB
records = [{'Product Name': 'Aalopuri', 'Revenue': '20', 'Demand': '1483', 'Initial Inventory': '10440.0'}, {'Product Name': 'Cold coffee', 'Revenue': '40', 'Demand': '1918', 'Initial Inventory': '13610.0'}, {'Product Name': 'Frankie', 'Revenue': '50', 'Demand': '1623', 'Initial Inventory': '11500.0'}, {'Product Name': 'Panipuri', 'Revenue': '20', 'Demand': '1720', 'Initial Inventory': '12260.0'}, {'Product Name': 'Sandwich', 'Revenue': '60', 'Demand': '1558', 'Initial Inventory': '10970.0'}, {'Product Name': 'Sugarcane juice', 'Revenue': '25', 'Demand': '1791', 'Initial Inventory': '12780.0'}, {'Product Name': 'Vadapav', 'Revenue': '20', 'Demand': '1426', 'Initial Inventory': '10060.0'}]
aalop_products = []
for rec in records:
    pname = rec['Product Name']
    if pname.lower().startswith('aalop'):
        aalop_products.append(pname)
if not aalop_products:
    raise ValueError("No products classified as 'Aalop' found in the data.")
revenue = {}
demand = {}
inventory = {}
for rec in records:
    pname = rec['Product Name']
    if pname in aalop_products:
        try:
            revenue[pname] = float(rec['Revenue'])
            demand[pname] = float(rec['Demand'])
            inventory[pname] = float(rec['Initial Inventory'])
        except Exception as e:
            raise ValueError(f'Invalid data for product {pname}: {e}')
for pname in aalop_products:
    if pname not in revenue or pname not in demand or pname not in inventory:
        raise ValueError(f'Missing data for product {pname}')
m = gp.Model('Aalop_Revenue_Maximization')
m.Params.MIPGap = 0.0001
x_vars = m.addVars(aalop_products, lb=0, vtype=GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((revenue[i] * x_vars[i] for i in aalop_products)), GRB.MAXIMIZE)
m.addConstrs((x_vars[i] <= demand[i] for i in aalop_products), name='')
m.addConstrs((x_vars[i] <= inventory[i] for i in aalop_products), name='')
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')