CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'A utility company must build a least-cost communication backbone connecting all substations in a region. '
          'The complete node set is listed in network_nodes.csv, the available undirected links and their construction '
          'costs are given in network_edges.csv, and the root node used for the flow-based connectivity formulation is '
          'given in network_parameters.csv.\n'
          '\n'
          'Formulate a minimum spanning tree network-design model. For each candidate link e, define y_e as a binary '
          'variable equal to 1 if link e is built and 0 otherwise. To enforce connectivity, define directed auxiliary '
          'flow variables f_ij on both directions of each candidate link, sending one unit of flow from the root to '
          'every other node. The objective is to minimize total construction cost. The model should select exactly n-1 '
          'links, where n is the number of nodes in network_nodes.csv, satisfy the flow-conservation connectivity '
          'constraints, link auxiliary flow to selected links, impose nonnegativity on flow variables, and impose '
          'binary restrictions on link-selection variables.',
 'relationships': [],
 'route': 'Others',
 'tables': [{'columns': ['Node'],
             'file_index': 0,
             'file_name': 'network_nodes.csv',
             'filters': {},
             'original_rows': 8,
             'records': [{'source_row': 0, 'values': {'Node': 'N1'}},
                         {'source_row': 1, 'values': {'Node': 'N2'}},
                         {'source_row': 2, 'values': {'Node': 'N3'}},
                         {'source_row': 3, 'values': {'Node': 'N4'}},
                         {'source_row': 4, 'values': {'Node': 'N5'}},
                         {'source_row': 5, 'values': {'Node': 'N6'}},
                         {'source_row': 6, 'values': {'Node': 'N7'}},
                         {'source_row': 7, 'values': {'Node': 'N8'}}],
             'returned_rows': 8,
             'role': 'descriptive non-unique role',
             'table_id': 'file_0_view_0'},
            {'columns': ['Node1', 'Node2', 'ConstructionCost'],
             'file_index': 1,
             'file_name': 'network_edges.csv',
             'filters': {},
             'original_rows': 14,
             'records': [{'source_row': 0, 'values': {'ConstructionCost': '4', 'Node1': 'N1', 'Node2': 'N2'}},
                         {'source_row': 1, 'values': {'ConstructionCost': '3', 'Node1': 'N1', 'Node2': 'N3'}},
                         {'source_row': 2, 'values': {'ConstructionCost': '9', 'Node1': 'N1', 'Node2': 'N4'}},
                         {'source_row': 3, 'values': {'ConstructionCost': '5', 'Node1': 'N2', 'Node2': 'N3'}},
                         {'source_row': 4, 'values': {'ConstructionCost': '6', 'Node1': 'N2', 'Node2': 'N5'}},
                         {'source_row': 5, 'values': {'ConstructionCost': '4', 'Node1': 'N3', 'Node2': 'N4'}},
                         {'source_row': 6, 'values': {'ConstructionCost': '7', 'Node1': 'N3', 'Node2': 'N6'}},
                         {'source_row': 7, 'values': {'ConstructionCost': '2', 'Node1': 'N4', 'Node2': 'N6'}},
                         {'source_row': 8, 'values': {'ConstructionCost': '8', 'Node1': 'N4', 'Node2': 'N7'}},
                         {'source_row': 9, 'values': {'ConstructionCost': '3', 'Node1': 'N5', 'Node2': 'N6'}},
                         {'source_row': 10, 'values': {'ConstructionCost': '10', 'Node1': 'N5', 'Node2': 'N8'}},
                         {'source_row': 11, 'values': {'ConstructionCost': '4', 'Node1': 'N6', 'Node2': 'N7'}},
                         {'source_row': 12, 'values': {'ConstructionCost': '6', 'Node1': 'N6', 'Node2': 'N8'}},
                         {'source_row': 13, 'values': {'ConstructionCost': '5', 'Node1': 'N7', 'Node2': 'N8'}}],
             'returned_rows': 14,
             'role': 'descriptive non-unique role',
             'table_id': 'file_1_view_0'},
            {'columns': ['Parameter', 'Value'],
             'file_index': 2,
             'file_name': 'network_parameters.csv',
             'filters': {},
             'original_rows': 1,
             'records': [{'source_row': 0, 'values': {'Parameter': 'RootNode', 'Value': 'N1'}}],
             'returned_rows': 1,
             'role': 'descriptive non-unique role',
             'table_id': 'file_2_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import gurobipy as gp
from gurobipy import GRB
node_records = CSVQA_DATA['tables'][0]['records']
nodes = [rec['values']['Node'] for rec in node_records]
edge_records = CSVQA_DATA['tables'][1]['records']
edge_costs = {}
edges = []
for rec in edge_records:
    n1 = rec['values']['Node1']
    n2 = rec['values']['Node2']
    cost = float(rec['values']['ConstructionCost'])
    edge_key = tuple(sorted((n1, n2)))
    edge_costs[edge_key] = cost
    edges.append((n1, n2, cost))
root = None
for rec in CSVQA_DATA['tables'][2]['records']:
    if rec['values']['Parameter'] == 'RootNode':
        root = rec['values']['Value']
        break
n = len(nodes)

def build_model():
    m = gp.Model()
    m.setParam('MIPGap', 1e-06)
    edge_selected = {ek: m.addVar(vtype=GRB.BINARY, name=f'y_{ek[0]}_{ek[1]}') for ek in edge_costs}
    directed_endpoints = {}
    edge_of_arc = {}
    flow = {}
    for (n1, n2, _) in edges:
        arc1 = (n1, n2)
        directed_endpoints[arc1] = (n1, n2)
        edge_of_arc[arc1] = tuple(sorted((n1, n2)))
        flow[arc1] = m.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f'f_{n1}_{n2}')
        arc2 = (n2, n1)
        directed_endpoints[arc2] = (n2, n1)
        edge_of_arc[arc2] = tuple(sorted((n1, n2)))
        flow[arc2] = m.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f'f_{n2}_{n1}')
    m.setObjective(gp.quicksum((edge_costs[ek] * edge_selected[ek] for ek in edge_costs)), GRB.MINIMIZE)
    m.addConstr(gp.quicksum((edge_selected[ek] for ek in edge_costs)) == n - 1, name='tree_size')
    net_supply = {node: -1 for node in nodes}
    net_supply[root] = n - 1
    for node in nodes:
        out = gp.quicksum((flow[arc] for arc in directed_endpoints if directed_endpoints[arc][0] == node))
        inn = gp.quicksum((flow[arc] for arc in directed_endpoints if directed_endpoints[arc][1] == node))
        m.addConstr(out - inn == net_supply[node], name=f'flow_conserve_{node}')
    for arc in directed_endpoints:
        ekey = edge_of_arc[arc]
        m.addConstr(flow[arc] <= (n - 1) * edge_selected[ekey], name=f'link_flow_{arc}')
    m.optimize()
    return m
m = build_model()