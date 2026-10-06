LEGACY_OBSERVATION = 'capacity.csv\nsection_light_inspections_last_year,section_signage_updates_last_year,SectionID,section_cleaning_minutes_last_month,aisle_signage_count,Capacity\n3,2,1,180,6,100\n4,4,2,300,4,150\n6,6,3,300,5,120\n8,4,4,180,5,130\n3,6,5,300,4,90\n6,3,6,300,5,110\n8,3,7,360,4,160\n6,6,8,300,4,140\n\nproducts.csv\nmerchandising_theme,packaging_label_review_count,marketing_campaign_format,ProductName,Value,product_catalog_page_views,supplier_contact_channel,Weight,supplier_catalog_revision_count\nFeatured,7,Newsletter,1,10,1040,Phone,2,4\nSeasonal,5,Brochure,2,15,340,Portal,3,6\nEveryday,7,Brochure,3,8,180,Portal,1,2\nSeasonal,5,Brochure,4,12,1380,Phone,2,4\nSeasonal,5,Brochure,5,20,1380,Portal,4,6\nEveryday,5,Newsletter,6,25,1040,Phone,5,1\nSeasonal,3,Web feature,7,5,560,Phone,1,1\nSeasonal,2,Web feature,8,30,790,Email,6,3\nSeasonal,3,Newsletter,9,18,340,Portal,3,3\nFeatured,5,Brochure,10,22,340,Phone,4,2'
LEGACY_RECORDS = [{'source': 'capacity.csv', 'values': {'section_light_inspections_last_year': '3', 'section_signage_updates_last_year': '2', 'SectionID': '1', 'section_cleaning_minutes_last_month': '180', 'aisle_signage_count': '6', 'Capacity': '100'}}, {'source': 'capacity.csv', 'values': {'section_light_inspections_last_year': '4', 'section_signage_updates_last_year': '4', 'SectionID': '2', 'section_cleaning_minutes_last_month': '300', 'aisle_signage_count': '4', 'Capacity': '150'}}, {'source': 'capacity.csv', 'values': {'section_light_inspections_last_year': '6', 'section_signage_updates_last_year': '6', 'SectionID': '3', 'section_cleaning_minutes_last_month': '300', 'aisle_signage_count': '5', 'Capacity': '120'}}, {'source': 'capacity.csv', 'values': {'section_light_inspections_last_year': '8', 'section_signage_updates_last_year': '4', 'SectionID': '4', 'section_cleaning_minutes_last_month': '180', 'aisle_signage_count': '5', 'Capacity': '130'}}, {'source': 'capacity.csv', 'values': {'section_light_inspections_last_year': '3', 'section_signage_updates_last_year': '6', 'SectionID': '5', 'section_cleaning_minutes_last_month': '300', 'aisle_signage_count': '4', 'Capacity': '90'}}, {'source': 'capacity.csv', 'values': {'section_light_inspections_last_year': '6', 'section_signage_updates_last_year': '3', 'SectionID': '6', 'section_cleaning_minutes_last_month': '300', 'aisle_signage_count': '5', 'Capacity': '110'}}, {'source': 'capacity.csv', 'values': {'section_light_inspections_last_year': '8', 'section_signage_updates_last_year': '3', 'SectionID': '7', 'section_cleaning_minutes_last_month': '360', 'aisle_signage_count': '4', 'Capacity': '160'}}, {'source': 'capacity.csv', 'values': {'section_light_inspections_last_year': '6', 'section_signage_updates_last_year': '6', 'SectionID': '8', 'section_cleaning_minutes_last_month': '300', 'aisle_signage_count': '4', 'Capacity': '140'}}, {'source': 'products.csv', 'values': {'merchandising_theme': 'Featured', 'packaging_label_review_count': '7', 'marketing_campaign_format': 'Newsletter', 'ProductName': '1', 'Value': '10', 'product_catalog_page_views': '1040', 'supplier_contact_channel': 'Phone', 'Weight': '2', 'supplier_catalog_revision_count': '4'}}, {'source': 'products.csv', 'values': {'merchandising_theme': 'Seasonal', 'packaging_label_review_count': '5', 'marketing_campaign_format': 'Brochure', 'ProductName': '2', 'Value': '15', 'product_catalog_page_views': '340', 'supplier_contact_channel': 'Portal', 'Weight': '3', 'supplier_catalog_revision_count': '6'}}, {'source': 'products.csv', 'values': {'merchandising_theme': 'Everyday', 'packaging_label_review_count': '7', 'marketing_campaign_format': 'Brochure', 'ProductName': '3', 'Value': '8', 'product_catalog_page_views': '180', 'supplier_contact_channel': 'Portal', 'Weight': '1', 'supplier_catalog_revision_count': '2'}}, {'source': 'products.csv', 'values': {'merchandising_theme': 'Seasonal', 'packaging_label_review_count': '5', 'marketing_campaign_format': 'Brochure', 'ProductName': '4', 'Value': '12', 'product_catalog_page_views': '1380', 'supplier_contact_channel': 'Phone', 'Weight': '2', 'supplier_catalog_revision_count': '4'}}, {'source': 'products.csv', 'values': {'merchandising_theme': 'Seasonal', 'packaging_label_review_count': '5', 'marketing_campaign_format': 'Brochure', 'ProductName': '5', 'Value': '20', 'product_catalog_page_views': '1380', 'supplier_contact_channel': 'Portal', 'Weight': '4', 'supplier_catalog_revision_count': '6'}}, {'source': 'products.csv', 'values': {'merchandising_theme': 'Everyday', 'packaging_label_review_count': '5', 'marketing_campaign_format': 'Newsletter', 'ProductName': '6', 'Value': '25', 'product_catalog_page_views': '1040', 'supplier_contact_channel': 'Phone', 'Weight': '5', 'supplier_catalog_revision_count': '1'}}, {'source': 'products.csv', 'values': {'merchandising_theme': 'Seasonal', 'packaging_label_review_count': '3', 'marketing_campaign_format': 'Web feature', 'ProductName': '7', 'Value': '5', 'product_catalog_page_views': '560', 'supplier_contact_channel': 'Phone', 'Weight': '1', 'supplier_catalog_revision_count': '1'}}, {'source': 'products.csv', 'values': {'merchandising_theme': 'Seasonal', 'packaging_label_review_count': '2', 'marketing_campaign_format': 'Web feature', 'ProductName': '8', 'Value': '30', 'product_catalog_page_views': '790', 'supplier_contact_channel': 'Email', 'Weight': '6', 'supplier_catalog_revision_count': '3'}}, {'source': 'products.csv', 'values': {'merchandising_theme': 'Seasonal', 'packaging_label_review_count': '3', 'marketing_campaign_format': 'Newsletter', 'ProductName': '9', 'Value': '18', 'product_catalog_page_views': '340', 'supplier_contact_channel': 'Portal', 'Weight': '3', 'supplier_catalog_revision_count': '3'}}, {'source': 'products.csv', 'values': {'merchandising_theme': 'Featured', 'packaging_label_review_count': '5', 'marketing_campaign_format': 'Brochure', 'ProductName': '10', 'Value': '22', 'product_catalog_page_views': '340', 'supplier_contact_channel': 'Phone', 'Weight': '4', 'supplier_catalog_revision_count': '2'}}]
import gurobipy as gp
from gurobipy import GRB
sections = []
capacities = {}
for rec in LEGACY_RECORDS:
    if rec['source'] == 'capacity.csv':
        sid = rec['values']['SectionID']
        sections.append(sid)
        capacities[sid] = int(rec['values']['Capacity'])
products = []
values = {}
weights = {}
for rec in LEGACY_RECORDS:
    if rec['source'] == 'products.csv':
        pid = rec['values']['ProductName']
        products.append(pid)
        values[pid] = int(rec['values']['Value'])
        weights[pid] = int(rec['values']['Weight'])
if len(sections) == 0 or len(products) == 0:
    raise ValueError('No sections or products found in LEGACY_RECORDS.')
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