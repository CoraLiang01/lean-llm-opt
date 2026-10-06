LEGACY_OBSERVATION = 'capacity.csv\nresource_id,customer_support_team_size,resource_capacity,platform_support_tickets_last_month\n1,20,1336,410\n2,20,1754,920\n3,12,1617,120\n4,12,1119,120\n5,24,1410,920\n6,8,627,250\n7,12,748,410\n8,24,1540,250\n9,8,1292,410\n10,16,1138,250\n\nproducts.csv\nitem_name,player_review_score,advertising_channel,item_value,community_forum_post_count,resource_requirement\nRacing,3.8,Video,28,650,393\nSports,4.7,Search,69,120,195\nAction,4.7,Video,20,420,192\nAdventure,4.4,Video,62,650,155\nRPG,3.8,Newsletter,58,980,500\nShooter,4.1,Newsletter,11,650,156\nStrategy,4.1,Video,73,120,317\nSimulation,3.8,Newsletter,43,240,694\nPuzzle,4.1,Video,28,240,751\nFighting,4.1,Newsletter,57,420,467\nPlatformer,3.2,Search,92,650,796\nSurvival,4.4,Search,66,420,146\nHorror,4.4,Video,14,240,269\nSandbox,3.8,Video,49,420,246\nMMO,3.8,Search,12,120,652'
LEGACY_RECORDS = [{'source': 'capacity.csv', 'values': {'resource_id': '1', 'customer_support_team_size': '20', 'resource_capacity': '1336', 'platform_support_tickets_last_month': '410'}}, {'source': 'capacity.csv', 'values': {'resource_id': '2', 'customer_support_team_size': '20', 'resource_capacity': '1754', 'platform_support_tickets_last_month': '920'}}, {'source': 'capacity.csv', 'values': {'resource_id': '3', 'customer_support_team_size': '12', 'resource_capacity': '1617', 'platform_support_tickets_last_month': '120'}}, {'source': 'capacity.csv', 'values': {'resource_id': '4', 'customer_support_team_size': '12', 'resource_capacity': '1119', 'platform_support_tickets_last_month': '120'}}, {'source': 'capacity.csv', 'values': {'resource_id': '5', 'customer_support_team_size': '24', 'resource_capacity': '1410', 'platform_support_tickets_last_month': '920'}}, {'source': 'capacity.csv', 'values': {'resource_id': '6', 'customer_support_team_size': '8', 'resource_capacity': '627', 'platform_support_tickets_last_month': '250'}}, {'source': 'capacity.csv', 'values': {'resource_id': '7', 'customer_support_team_size': '12', 'resource_capacity': '748', 'platform_support_tickets_last_month': '410'}}, {'source': 'capacity.csv', 'values': {'resource_id': '8', 'customer_support_team_size': '24', 'resource_capacity': '1540', 'platform_support_tickets_last_month': '250'}}, {'source': 'capacity.csv', 'values': {'resource_id': '9', 'customer_support_team_size': '8', 'resource_capacity': '1292', 'platform_support_tickets_last_month': '410'}}, {'source': 'capacity.csv', 'values': {'resource_id': '10', 'customer_support_team_size': '16', 'resource_capacity': '1138', 'platform_support_tickets_last_month': '250'}}, {'source': 'products.csv', 'values': {'item_name': 'Racing', 'player_review_score': '3.8', 'advertising_channel': 'Video', 'item_value': '28', 'community_forum_post_count': '650', 'resource_requirement': '393'}}, {'source': 'products.csv', 'values': {'item_name': 'Sports', 'player_review_score': '4.7', 'advertising_channel': 'Search', 'item_value': '69', 'community_forum_post_count': '120', 'resource_requirement': '195'}}, {'source': 'products.csv', 'values': {'item_name': 'Action', 'player_review_score': '4.7', 'advertising_channel': 'Video', 'item_value': '20', 'community_forum_post_count': '420', 'resource_requirement': '192'}}, {'source': 'products.csv', 'values': {'item_name': 'Adventure', 'player_review_score': '4.4', 'advertising_channel': 'Video', 'item_value': '62', 'community_forum_post_count': '650', 'resource_requirement': '155'}}, {'source': 'products.csv', 'values': {'item_name': 'RPG', 'player_review_score': '3.8', 'advertising_channel': 'Newsletter', 'item_value': '58', 'community_forum_post_count': '980', 'resource_requirement': '500'}}, {'source': 'products.csv', 'values': {'item_name': 'Shooter', 'player_review_score': '4.1', 'advertising_channel': 'Newsletter', 'item_value': '11', 'community_forum_post_count': '650', 'resource_requirement': '156'}}, {'source': 'products.csv', 'values': {'item_name': 'Strategy', 'player_review_score': '4.1', 'advertising_channel': 'Video', 'item_value': '73', 'community_forum_post_count': '120', 'resource_requirement': '317'}}, {'source': 'products.csv', 'values': {'item_name': 'Simulation', 'player_review_score': '3.8', 'advertising_channel': 'Newsletter', 'item_value': '43', 'community_forum_post_count': '240', 'resource_requirement': '694'}}, {'source': 'products.csv', 'values': {'item_name': 'Puzzle', 'player_review_score': '4.1', 'advertising_channel': 'Video', 'item_value': '28', 'community_forum_post_count': '240', 'resource_requirement': '751'}}, {'source': 'products.csv', 'values': {'item_name': 'Fighting', 'player_review_score': '4.1', 'advertising_channel': 'Newsletter', 'item_value': '57', 'community_forum_post_count': '420', 'resource_requirement': '467'}}, {'source': 'products.csv', 'values': {'item_name': 'Platformer', 'player_review_score': '3.2', 'advertising_channel': 'Search', 'item_value': '92', 'community_forum_post_count': '650', 'resource_requirement': '796'}}, {'source': 'products.csv', 'values': {'item_name': 'Survival', 'player_review_score': '4.4', 'advertising_channel': 'Search', 'item_value': '66', 'community_forum_post_count': '420', 'resource_requirement': '146'}}, {'source': 'products.csv', 'values': {'item_name': 'Horror', 'player_review_score': '4.4', 'advertising_channel': 'Video', 'item_value': '14', 'community_forum_post_count': '240', 'resource_requirement': '269'}}, {'source': 'products.csv', 'values': {'item_name': 'Sandbox', 'player_review_score': '3.8', 'advertising_channel': 'Video', 'item_value': '49', 'community_forum_post_count': '420', 'resource_requirement': '246'}}, {'source': 'products.csv', 'values': {'item_name': 'MMO', 'player_review_score': '3.8', 'advertising_channel': 'Search', 'item_value': '12', 'community_forum_post_count': '120', 'resource_requirement': '652'}}]
import gurobipy as gp
from gurobipy import GRB
platforms = []
capacities = {}
genres = []
values = {}
requirements = {}
for rec in LEGACY_RECORDS:
    if rec['source'] == 'capacity.csv':
        rid = rec['values']['resource_id']
        platforms.append(rid)
        capacities[rid] = int(rec['values']['resource_capacity'])
    elif rec['source'] == 'products.csv':
        name = rec['values']['item_name']
        genres.append(name)
        values[name] = int(rec['values']['item_value'])
        requirements[name] = int(rec['values']['resource_requirement'])
if len(platforms) == 0 or len(genres) == 0:
    raise ValueError('Missing platforms or genres in LEGACY_RECORDS')
for rid in platforms:
    if rid not in capacities:
        raise ValueError(f'Missing capacity for platform {rid}')
for name in genres:
    if name not in values or name not in requirements:
        raise ValueError(f'Missing value or requirement for genre {name}')
m = gp.Model('Game_Listing_Optimization')
x = m.addVars(platforms, genres, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((values[j] * x[i, j] for i in platforms for j in genres)), GRB.MAXIMIZE)
m.addConstrs((gp.quicksum((requirements[j] * x[i, j] for j in genres)) <= capacities[i] for i in platforms), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')