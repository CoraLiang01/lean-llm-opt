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

def build_model():
    tables = CSVQA_DATA['tables']
    node_table = next((t for t in tables if t['table_id'] == 'file_0_view_0'))
    edge_table = next((t for t in tables if t['table_id'] == 'file_1_view_0'))
    param_table = next((t for t in tables if t['table_id'] == 'file_2_view_0'))
    node_ids = [rec['values']['Node'] for rec in node_table['records']]
    n = len(node_ids)
    edge_keys = []
    edge_cost = {}
    for rec in edge_table['records']:
        node1 = rec['values']['Node1']
        node2 = rec['values']['Node2']
        key = (node1, node2)
        edge_keys.append(key)
        edge_cost[key] = float(rec['values']['ConstructionCost'])
    root = None
    for rec in param_table['records']:
        if rec['values']['Parameter'] == 'RootNode':
            root = rec['values']['Value']
            break
    if root is None:
        raise ValueError('Root node not found')
    directed_endpoints = {}
    edge_of_arc = {}
    for key in edge_keys:
        i, j = key
        arc1 = (i, j)
        arc2 = (j, i)
        directed_endpoints[arc1] = (i, j)
        directed_endpoints[arc2] = (j, i)
        edge_of_arc[arc1] = key
        edge_of_arc[arc2] = key
    m = gp.Model()
    y = {e: m.addVar(vtype=GRB.BINARY, name=f'y_{e[0]}_{e[1]}') for e in edge_keys}
    f = {arc: m.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f'f_{arc[0]}_{arc[1]}') for arc in directed_endpoints}
    m.setObjective(gp.quicksum((edge_cost[e] * y[e] for e in edge_keys)), GRB.MINIMIZE)
    m.addConstr(gp.quicksum((y[e] for e in edge_keys)) == n - 1, name='edge_count')
    for arc, (tail, head) in directed_endpoints.items():
        e = edge_of_arc[arc]
        m.addConstr(f[arc] <= (n - 1) * y[e], name=f'flow_cap_{arc[0]}_{arc[1]}')
    outgoing = {node: [] for node in node_ids}
    incoming = {node: [] for node in node_ids}
    for arc, (tail, head) in directed_endpoints.items():
        outgoing[tail].append(f[arc])
        incoming[head].append(f[arc])
    net_supply = {node: -1 for node in node_ids}
    net_supply[root] = n - 1
    for node in node_ids:
        m.addConstr(gp.quicksum(outgoing[node]) - gp.quicksum(incoming[node]) == net_supply[node], name=f'flow_conserve_{node}')
    m.setParam('MIPGap', 1e-06)
    m.optimize()
    return m
m = build_model()