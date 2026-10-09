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
             'role': 'substation node list',
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
             'role': 'candidate undirected links and costs',
             'table_id': 'file_1_view_0'},
            {'columns': ['Parameter', 'Value'],
             'file_index': 2,
             'file_name': 'network_parameters.csv',
             'filters': {'conditions': [{'column': 'Parameter',
                                         'dtype': 'string',
                                         'evidence': 'the root node used for the flow-based connectivity formulation '
                                                     'is given in network_parameters.csv',
                                         'operator': 'exact',
                                         'value': 'RootNode'}],
                         'logic': 'and'},
             'original_rows': 1,
             'records': [{'source_row': 0, 'values': {'Parameter': 'RootNode', 'Value': 'N1'}}],
             'returned_rows': 1,
             'role': 'root node parameter',
             'table_id': 'file_2_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import pandas as pd
CSVQA_FRAMES = {t["table_id"]: pd.DataFrame([r["values"] for r in t["records"]], columns=t["columns"], index=[r["source_row"] for r in t["records"]]) for t in CSVQA_DATA["tables"]}
import gurobipy as gp
import pandas as pd

def solve_problem(CSVQA_FRAMES):
    nodes_frame = CSVQA_FRAMES['file_0_view_0']
    nodes = []
    for (_, row) in nodes_frame.iterrows():
        node = row['Node']
        if node in nodes:
            raise ValueError(f'Duplicate node: {node}')
        nodes.append(node)
    n_nodes = len(nodes)
    if n_nodes < 2:
        raise ValueError('At least two nodes required.')
    edges_frame = CSVQA_FRAMES['file_1_view_0']
    edge_keys = []
    edge_cost = {}
    for (_, row) in edges_frame.iterrows():
        i = row['Node1']
        j = row['Node2']
        if not (i in nodes and j in nodes):
            raise ValueError(f'Edge ({i},{j}) contains unknown node.')
        key = (i, j)
        if key in edge_cost:
            raise ValueError(f'Duplicate edge: {key}')
        try:
            cost = float(row['ConstructionCost'])
        except Exception:
            raise ValueError(f"Invalid ConstructionCost for edge ({i},{j}): {row['ConstructionCost']}")
        edge_keys.append(key)
        edge_cost[key] = cost
    param_frame = CSVQA_FRAMES['file_2_view_0']
    root_row = None
    for (_, row) in param_frame.iterrows():
        if str(row['Parameter']).casefold() == 'rootnode':
            root_row = row
            break
    if root_row is None:
        raise ValueError('RootNode parameter not found in network_parameters.csv')
    root = root_row['Value']
    if root not in nodes:
        raise ValueError(f'Root node {root} not in node list.')
    arc_keys = []
    for (i, j) in edge_keys:
        arc_keys.append((i, j))
        arc_keys.append((j, i))
    neighbors = {v: set() for v in nodes}
    for (i, j) in edge_keys:
        neighbors[i].add(j)
        neighbors[j].add(i)
    m = gp.Model('MinimumSpanningTreeNetworkDesign')
    y_vars = m.addVars(edge_keys, vtype=gp.GRB.BINARY, name='')
    f_vars = m.addVars(arc_keys, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
    m.setObjective(gp.quicksum((edge_cost[e] * y_vars[e] for e in edge_keys)), gp.GRB.MINIMIZE)
    m.addConstr(y_vars.sum() == n_nodes - 1, name='spanning_tree_link_count')
    for v in nodes:
        out_arcs = []
        in_arcs = []
        for u in neighbors[v]:
            if (v, u) in f_vars:
                out_arcs.append(f_vars[v, u])
            if (u, v) in f_vars:
                in_arcs.append(f_vars[u, v])
        if v == root:
            m.addConstr(gp.quicksum(out_arcs) - gp.quicksum(in_arcs) == n_nodes - 1, name=f'flow_conserv_root_{v}')
        else:
            m.addConstr(gp.quicksum(out_arcs) - gp.quicksum(in_arcs) == -1, name=f'flow_conserv_{v}')
    for (i, j) in edge_keys:
        m.addConstr(f_vars[i, j] <= (n_nodes - 1) * y_vars[i, j], name=f'flow_link_{i}_{j}_fij')
        m.addConstr(f_vars[j, i] <= (n_nodes - 1) * y_vars[i, j], name=f'flow_link_{i}_{j}_fji')
    m.Params.MIPGap = 0.0001
    m.optimize()
    return m
m = solve_problem(CSVQA_FRAMES)
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal {m.objVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')