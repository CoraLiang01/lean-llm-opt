LEGACY_OBSERVATION = 'products.csv\nProductName,marketing_campaign_format,Weight,supplier_contact_channel,packaging_label_review_count,supplier_catalog_revision_count,supplier_quality_rating,Value,catalog_display_group\nSpinach,Brochure,230,Email,3,3,3.2,64,Featured\nShiitake Mushrooms,Brochure,637,Portal,5,3,4.1,75,Featured\nApples,Brochure,773,Email,3,1,4.7,68,Seasonal\nCarrots,Brochure,653,Portal,2,3,4.7,11,Featured\nBasil,Web feature,755,Email,3,6,4.1,91,Seasonal\nPotatoes,Brochure,670,Portal,3,6,3.5,31,Everyday\nGreen Beans,Brochure,505,Phone,9,1,4.1,90,Featured\nBlueberries,Brochure,821,Portal,5,6,3.5,56,Seasonal\nOranges,Web feature,83,Phone,3,4,3.2,10,Seasonal\nWatermelons,Web feature,249,Email,7,3,4.4,24,Everyday\n\ncapacity.csv\nrefrigeration_service_visits,Capacity,storage_sanitation_inspections_last_year\n8,875,6'
LEGACY_RECORDS = [{'source': 'products.csv', 'values': {'ProductName': 'Spinach', 'marketing_campaign_format': 'Brochure', 'Weight': '230', 'supplier_contact_channel': 'Email', 'packaging_label_review_count': '3', 'supplier_catalog_revision_count': '3', 'supplier_quality_rating': '3.2', 'Value': '64', 'catalog_display_group': 'Featured'}}, {'source': 'products.csv', 'values': {'ProductName': 'Shiitake Mushrooms', 'marketing_campaign_format': 'Brochure', 'Weight': '637', 'supplier_contact_channel': 'Portal', 'packaging_label_review_count': '5', 'supplier_catalog_revision_count': '3', 'supplier_quality_rating': '4.1', 'Value': '75', 'catalog_display_group': 'Featured'}}, {'source': 'products.csv', 'values': {'ProductName': 'Apples', 'marketing_campaign_format': 'Brochure', 'Weight': '773', 'supplier_contact_channel': 'Email', 'packaging_label_review_count': '3', 'supplier_catalog_revision_count': '1', 'supplier_quality_rating': '4.7', 'Value': '68', 'catalog_display_group': 'Seasonal'}}, {'source': 'products.csv', 'values': {'ProductName': 'Carrots', 'marketing_campaign_format': 'Brochure', 'Weight': '653', 'supplier_contact_channel': 'Portal', 'packaging_label_review_count': '2', 'supplier_catalog_revision_count': '3', 'supplier_quality_rating': '4.7', 'Value': '11', 'catalog_display_group': 'Featured'}}, {'source': 'products.csv', 'values': {'ProductName': 'Basil', 'marketing_campaign_format': 'Web feature', 'Weight': '755', 'supplier_contact_channel': 'Email', 'packaging_label_review_count': '3', 'supplier_catalog_revision_count': '6', 'supplier_quality_rating': '4.1', 'Value': '91', 'catalog_display_group': 'Seasonal'}}, {'source': 'products.csv', 'values': {'ProductName': 'Potatoes', 'marketing_campaign_format': 'Brochure', 'Weight': '670', 'supplier_contact_channel': 'Portal', 'packaging_label_review_count': '3', 'supplier_catalog_revision_count': '6', 'supplier_quality_rating': '3.5', 'Value': '31', 'catalog_display_group': 'Everyday'}}, {'source': 'products.csv', 'values': {'ProductName': 'Green Beans', 'marketing_campaign_format': 'Brochure', 'Weight': '505', 'supplier_contact_channel': 'Phone', 'packaging_label_review_count': '9', 'supplier_catalog_revision_count': '1', 'supplier_quality_rating': '4.1', 'Value': '90', 'catalog_display_group': 'Featured'}}, {'source': 'products.csv', 'values': {'ProductName': 'Blueberries', 'marketing_campaign_format': 'Brochure', 'Weight': '821', 'supplier_contact_channel': 'Portal', 'packaging_label_review_count': '5', 'supplier_catalog_revision_count': '6', 'supplier_quality_rating': '3.5', 'Value': '56', 'catalog_display_group': 'Seasonal'}}, {'source': 'products.csv', 'values': {'ProductName': 'Oranges', 'marketing_campaign_format': 'Web feature', 'Weight': '83', 'supplier_contact_channel': 'Phone', 'packaging_label_review_count': '3', 'supplier_catalog_revision_count': '4', 'supplier_quality_rating': '3.2', 'Value': '10', 'catalog_display_group': 'Seasonal'}}, {'source': 'products.csv', 'values': {'ProductName': 'Watermelons', 'marketing_campaign_format': 'Web feature', 'Weight': '249', 'supplier_contact_channel': 'Email', 'packaging_label_review_count': '7', 'supplier_catalog_revision_count': '3', 'supplier_quality_rating': '4.4', 'Value': '24', 'catalog_display_group': 'Everyday'}}, {'source': 'capacity.csv', 'values': {'refrigeration_service_visits': '8', 'Capacity': '875', 'storage_sanitation_inspections_last_year': '6'}}]
import gurobipy as gp
from gurobipy import GRB
products = []
value = {}
weight = {}
for rec in LEGACY_RECORDS:
    if rec['source'] == 'products.csv':
        pname = rec['values']['ProductName']
        products.append(pname)
        value[pname] = int(rec['values']['Value'])
        weight[pname] = int(rec['values']['Weight'])
capacities = []
for rec in LEGACY_RECORDS:
    if rec['source'] == 'capacity.csv':
        for k, v in rec['values'].items():
            if k.lower() == 'capacity':
                capacities.append(int(v))
if not capacities:
    raise ValueError('No capacity found in LEGACY_RECORDS')
total_capacity = sum(capacities)
for pname in products:
    if pname not in value or pname not in weight:
        raise ValueError(f'Missing value or weight for product {pname}')
m = gp.Model('Supermarket_Stock')
x = m.addVars(products, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((value[p] * x[p] for p in products)), GRB.MAXIMIZE)
m.addConstr(gp.quicksum((weight[p] * x[p] for p in products)) <= total_capacity, name='capacity')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')