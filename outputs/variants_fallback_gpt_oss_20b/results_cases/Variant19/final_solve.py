CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'A communications team must build a connected backbone over all listed sites. The node set is listed in '
          'network_nodes.csv, available undirected links and construction costs are listed in network_edges.csv, and '
          'the root node for a single-commodity flow formulation is listed in network_parameters.csv.\n'
          '\n'
          'Formulate a minimum-cost spanning tree model. For each undirected edge i-j, define y_ij as a binary '
          'variable equal to 1 if the edge is selected. For each directed arc version of an available edge, define '
          'f_ij as the nonnegative connectivity flow sent from the root. The objective is to minimize selected edge '
          'construction cost. The model should include a constraint selecting exactly n-1 edges, root and non-root '
          'flow-balance constraints using the node set in network_nodes.csv, flow-to-edge linking constraints with M = '
          'n-1, nonnegativity constraints for flow variables, and binary restrictions for edge-selection variables.',
 'relationships': [],
 'route': 'Others',
 'tables': [{'columns': ['Node'],
             'file_index': 0,
             'file_name': 'network_nodes.csv',
             'filters': {},
             'original_rows': 9,
             'records': [{'source_row': 0, 'values': {'Node': 'V1'}},
                         {'source_row': 1, 'values': {'Node': 'V2'}},
                         {'source_row': 2, 'values': {'Node': 'V3'}},
                         {'source_row': 3, 'values': {'Node': 'V4'}},
                         {'source_row': 4, 'values': {'Node': 'V5'}},
                         {'source_row': 5, 'values': {'Node': 'V6'}},
                         {'source_row': 6, 'values': {'Node': 'V7'}},
                         {'source_row': 7, 'values': {'Node': 'V8'}},
                         {'source_row': 8, 'values': {'Node': 'V9'}}],
             'returned_rows': 9,
             'role': 'descriptive non-unique role',
             'table_id': 'file_0_view_0'},
            {'columns': ['Node1', 'Node2', 'ConstructionCost'],
             'file_index': 1,
             'file_name': 'network_edges.csv',
             'filters': {},
             'original_rows': 16,
             'records': [{'source_row': 0, 'values': {'ConstructionCost': '6', 'Node1': 'V1', 'Node2': 'V2'}},
                         {'source_row': 1, 'values': {'ConstructionCost': '4', 'Node1': 'V1', 'Node2': 'V3'}},
                         {'source_row': 2, 'values': {'ConstructionCost': '9', 'Node1': 'V1', 'Node2': 'V4'}},
                         {'source_row': 3, 'values': {'ConstructionCost': '5', 'Node1': 'V2', 'Node2': 'V3'}},
                         {'source_row': 4, 'values': {'ConstructionCost': '7', 'Node1': 'V2', 'Node2': 'V5'}},
                         {'source_row': 5, 'values': {'ConstructionCost': '3', 'Node1': 'V3', 'Node2': 'V4'}},
                         {'source_row': 6, 'values': {'ConstructionCost': '8', 'Node1': 'V3', 'Node2': 'V6'}},
                         {'source_row': 7, 'values': {'ConstructionCost': '4', 'Node1': 'V4', 'Node2': 'V6'}},
                         {'source_row': 8, 'values': {'ConstructionCost': '10', 'Node1': 'V4', 'Node2': 'V7'}},
                         {'source_row': 9, 'values': {'ConstructionCost': '6', 'Node1': 'V5', 'Node2': 'V6'}},
                         {'source_row': 10, 'values': {'ConstructionCost': '11', 'Node1': 'V5', 'Node2': 'V8'}},
                         {'source_row': 11, 'values': {'ConstructionCost': '5', 'Node1': 'V6', 'Node2': 'V7'}},
                         {'source_row': 12, 'values': {'ConstructionCost': '7', 'Node1': 'V6', 'Node2': 'V8'}},
                         {'source_row': 13, 'values': {'ConstructionCost': '4', 'Node1': 'V7', 'Node2': 'V8'}},
                         {'source_row': 14, 'values': {'ConstructionCost': '6', 'Node1': 'V7', 'Node2': 'V9'}},
                         {'source_row': 15, 'values': {'ConstructionCost': '5', 'Node1': 'V8', 'Node2': 'V9'}}],
             'returned_rows': 16,
             'role': 'descriptive non-unique role',
             'table_id': 'file_1_view_0'},
            {'columns': ['Parameter', 'Value'],
             'file_index': 2,
             'file_name': 'network_parameters.csv',
             'filters': {},
             'original_rows': 1,
             'records': [{'source_row': 0, 'values': {'Parameter': 'RootNode', 'Value': 'V1'}}],
             'returned_rows': 1,
             'role': 'descriptive non-unique role',
             'table_id': 'file_2_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import gurobipy
from gurobipy import GRB
tables = CSVQA_DATA['tables']

def get_table(table_id):
    for t in tables:
        if t['table_id'] == table_id:
            return t
    raise KeyError(f'Table {table_id} not found')
node_table = get_table('file_0_view_0')
node_ids = [rec['values']['Node'] for rec in node_table['records']]
edge_table = get_table('file_1_view_0')
edge_keys = []
edge_cost = {}
for rec in edge_table['records']:
    i = rec['values']['Node1']
    j = rec['values']['Node2']
    key = (i, j)
    edge_keys.append(key)
    edge_cost[key] = float(rec['values']['ConstructionCost'])
param_table = get_table('file_2_view_0')
root = None
for rec in param_table['records']:
    if rec['values']['Parameter'] == 'RootNode':
        root = rec['values']['Value']
        break
if root is None:
    raise ValueError('RootNode parameter not found')
n = len(node_ids)
M = n - 1
directed_endpoints = {}
edge_of_arc = {}
for e in edge_keys:
    (i, j) = e
    arc1 = (i, j)
    arc2 = (j, i)
    directed_endpoints[arc1] = (i, j)
    directed_endpoints[arc2] = (j, i)
    edge_of_arc[arc1] = e
    edge_of_arc[arc2] = e
m = gurobipy.Model()
m.Params.MIPGap = 1e-06
y_e = {e: m.addVar(vtype=GRB.BINARY, name=f'y_{e[0]}_{e[1]}') for e in edge_keys}
f_a = {a: m.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f'f_{a[0]}_{a[1]}') for a in directed_endpoints}
outgoing = {node: [] for node in node_ids}
incoming = {node: [] for node in node_ids}
for (arc, (tail, head)) in directed_endpoints.items():
    outgoing[tail].append(f_a[arc])
    incoming[head].append(f_a[arc])
    m.addConstr(f_a[arc] <= M * y_e[edge_of_arc[arc]])
net_supply = {node: -1 for node in node_ids}
net_supply[root] = M
for node in node_ids:
    m.addConstr(gurobipy.quicksum(outgoing[node]) - gurobipy.quicksum(incoming[node]) == net_supply[node])
m.addConstr(gurobipy.quicksum((y_e[e] for e in edge_keys)) == M)
m.setObjective(gurobipy.quicksum((edge_cost[e] * y_e[e] for e in edge_keys)), GRB.MINIMIZE)
m.optimize()