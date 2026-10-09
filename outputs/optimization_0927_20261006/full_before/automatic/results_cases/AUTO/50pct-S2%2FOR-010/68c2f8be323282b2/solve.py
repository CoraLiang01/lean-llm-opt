LEGACY_OBSERVATION = '{"values": {"SectionID": "1", "aisle_signage_count": "6", "Capacity": "100"}}\n{"values": {"SectionID": "2", "aisle_signage_count": "4", "Capacity": "150"}}\n{"values": {"SectionID": "3", "aisle_signage_count": "5", "Capacity": "120"}}\n{"values": {"SectionID": "4", "aisle_signage_count": "5", "Capacity": "130"}}\n{"values": {"SectionID": "5", "aisle_signage_count": "4", "Capacity": "90"}}\n{"values": {"SectionID": "6", "aisle_signage_count": "5", "Capacity": "110"}}\n{"values": {"SectionID": "7", "aisle_signage_count": "4", "Capacity": "160"}}\n{"values": {"SectionID": "8", "aisle_signage_count": "4", "Capacity": "140"}}\n{"values": {"merchandising_theme": "Featured", "ProductName": "1", "Value": "10", "product_catalog_page_views": "1040", "Weight": "2"}}\n{"values": {"merchandising_theme": "Seasonal", "ProductName": "2", "Value": "15", "product_catalog_page_views": "340", "Weight": "3"}}\n{"values": {"merchandising_theme": "Everyday", "ProductName": "3", "Value": "8", "product_catalog_page_views": "180", "Weight": "1"}}\n{"values": {"merchandising_theme": "Seasonal", "ProductName": "4", "Value": "12", "product_catalog_page_views": "1380", "Weight": "2"}}\n{"values": {"merchandising_theme": "Seasonal", "ProductName": "5", "Value": "20", "product_catalog_page_views": "1380", "Weight": "4"}}\n{"values": {"merchandising_theme": "Everyday", "ProductName": "6", "Value": "25", "product_catalog_page_views": "1040", "Weight": "5"}}\n{"values": {"merchandising_theme": "Seasonal", "ProductName": "7", "Value": "5", "product_catalog_page_views": "560", "Weight": "1"}}\n{"values": {"merchandising_theme": "Seasonal", "ProductName": "8", "Value": "30", "product_catalog_page_views": "790", "Weight": "6"}}\n{"values": {"merchandising_theme": "Seasonal", "ProductName": "9", "Value": "18", "product_catalog_page_views": "340", "Weight": "3"}}\n{"values": {"merchandising_theme": "Featured", "ProductName": "10", "Value": "22", "product_catalog_page_views": "340", "Weight": "4"}}'
LEGACY_RECORDS = [{'source': '', 'values': {'SectionID': '1', 'aisle_signage_count': '6', 'Capacity': '100'}}, {'source': '', 'values': {'SectionID': '2', 'aisle_signage_count': '4', 'Capacity': '150'}}, {'source': '', 'values': {'SectionID': '3', 'aisle_signage_count': '5', 'Capacity': '120'}}, {'source': '', 'values': {'SectionID': '4', 'aisle_signage_count': '5', 'Capacity': '130'}}, {'source': '', 'values': {'SectionID': '5', 'aisle_signage_count': '4', 'Capacity': '90'}}, {'source': '', 'values': {'SectionID': '6', 'aisle_signage_count': '5', 'Capacity': '110'}}, {'source': '', 'values': {'SectionID': '7', 'aisle_signage_count': '4', 'Capacity': '160'}}, {'source': '', 'values': {'SectionID': '8', 'aisle_signage_count': '4', 'Capacity': '140'}}, {'source': '', 'values': {'merchandising_theme': 'Featured', 'ProductName': '1', 'Value': '10', 'product_catalog_page_views': '1040', 'Weight': '2'}}, {'source': '', 'values': {'merchandising_theme': 'Seasonal', 'ProductName': '2', 'Value': '15', 'product_catalog_page_views': '340', 'Weight': '3'}}, {'source': '', 'values': {'merchandising_theme': 'Everyday', 'ProductName': '3', 'Value': '8', 'product_catalog_page_views': '180', 'Weight': '1'}}, {'source': '', 'values': {'merchandising_theme': 'Seasonal', 'ProductName': '4', 'Value': '12', 'product_catalog_page_views': '1380', 'Weight': '2'}}, {'source': '', 'values': {'merchandising_theme': 'Seasonal', 'ProductName': '5', 'Value': '20', 'product_catalog_page_views': '1380', 'Weight': '4'}}, {'source': '', 'values': {'merchandising_theme': 'Everyday', 'ProductName': '6', 'Value': '25', 'product_catalog_page_views': '1040', 'Weight': '5'}}, {'source': '', 'values': {'merchandising_theme': 'Seasonal', 'ProductName': '7', 'Value': '5', 'product_catalog_page_views': '560', 'Weight': '1'}}, {'source': '', 'values': {'merchandising_theme': 'Seasonal', 'ProductName': '8', 'Value': '30', 'product_catalog_page_views': '790', 'Weight': '6'}}, {'source': '', 'values': {'merchandising_theme': 'Seasonal', 'ProductName': '9', 'Value': '18', 'product_catalog_page_views': '340', 'Weight': '3'}}, {'source': '', 'values': {'merchandising_theme': 'Featured', 'ProductName': '10', 'Value': '22', 'product_catalog_page_views': '340', 'Weight': '4'}}]
import gurobipy as gp
from gurobipy import GRB
sections = []
capacities = {}
products = []
values = {}
weights = {}
for rec in LEGACY_RECORDS:
    v = rec['values']
    if 'SectionID' in v and 'Capacity' in v:
        sid = v['SectionID']
        sections.append(sid)
        capacities[sid] = int(v['Capacity'])
    if 'ProductName' in v and 'Value' in v and ('Weight' in v):
        pid = v['ProductName']
        products.append(pid)
        values[pid] = int(v['Value'])
        weights[pid] = int(v['Weight'])
if len(sections) == 0 or len(products) == 0:
    raise ValueError('Missing section or product data.')
for sid in sections:
    if sid not in capacities:
        raise ValueError(f'Missing capacity for section {sid}.')
for pid in products:
    if pid not in values or pid not in weights:
        raise ValueError(f'Missing value or weight for product {pid}.')
m = gp.Model('Supermarket_Section_Allocation')
x = m.addVars(sections, products, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((values[pid] * x[sid, pid] for sid in sections for pid in products)), GRB.MAXIMIZE)
m.addConstrs((gp.quicksum((weights[pid] * x[sid, pid] for pid in products)) <= capacities[sid] for sid in sections), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')