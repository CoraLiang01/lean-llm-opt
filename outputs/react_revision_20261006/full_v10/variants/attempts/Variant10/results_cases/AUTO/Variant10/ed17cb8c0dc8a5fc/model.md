Minimum Spanning Tree Network-Design Model

Sets
N: set of nodes (from file_0_view_0, column Node)
E: set of undirected edges (from file_1_view_0, columns Node1, Node2)
A: set of directed arcs, i.e., for each {i,j} ∈ E, both (i,j) and (j,i)
r: root node (from file_2_view_0, Parameter = RootNode, Value)

Parameters
c_{ij}: construction cost for undirected edge {i,j} ∈ E (from file_1_view_0, ConstructionCost)

Decision Variables
y_{ij} ∈ {0,1} for each {i,j} ∈ E (build edge {i,j})
f_{uv} ≥ 0 for each (u,v) ∈ A (auxiliary flow on arc (u,v))

Objective
minimize ∑_{ {i,j} ∈ E } c_{ij} y_{ij}

Constraints

1. Link count:
  ∑_{ {i,j} ∈ E } y_{ij} = |N| - 1

2. Flow conservation (for all k ∈ N, k ≠ r):
  ∑_{ (i,k) ∈ A } f_{ik} - ∑_{ (k,j) ∈ A } f_{kj} = 1

  For the root node r:
  ∑_{ (i,r) ∈ A } f_{ir} - ∑_{ (r,j) ∈ A } f_{rj} = 1 - (|N| - 1) = -( |N| - 2 )

3. Flow-edge linking (for all (u,v) ∈ A, where {u,v} ∈ E):
  f_{uv} ≤ (|N| - 1) y_{uv}
  (where y_{uv} = y_{vu} for undirected edge {u,v})

4. Variable domains:
  y_{ij} ∈ {0,1} for all {i,j} ∈ E
  f_{uv} ≥ 0 for all (u,v) ∈ A

Data Mapping

- Nodes N: file_0_view_0, column Node
- Edges E: file_1_view_0, columns Node1, Node2
- Construction costs c_{ij}: file_1_view_0, column ConstructionCost
- Root node r: file_2_view_0, Parameter = RootNode, Value
- Directed arcs A: for each {i,j} ∈ E, both (i,j) and (j,i)
- |N|: number of rows in file_0_view_0

All indices, parameters, and variables are defined over the full set of entities as listed in the current data.