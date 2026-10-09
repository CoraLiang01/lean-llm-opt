Mathematical Model

Sets:
- $S$: set of sources (from file_0_view_0, NodeType = SourceSupply)
- $C$: set of customers (from file_0_view_0, NodeType = CustomerDemand)
- $H$: set of hubs (from file_1_view_0, Hub)
- $A$: set of directed arcs $(i,j)$ with cost $c_{ij}$ (from file_2_view_0, (From, To, Cost))

Parameters:
- $s_i$: supply at source $i\in S$ (from file_0_view_0, Amount, NodeType = SourceSupply)
- $d_j$: demand at customer $j\in C$ (from file_0_view_0, Amount, NodeType = CustomerDemand)
- $u_h$: throughput capacity at hub $h\in H$ (from file_1_view_0, ThroughputCapacity)
- $c_{ij}$: per-unit cost on arc $(i,j)\in A$ (from file_2_view_0, Cost)
- $f_{ij}$: nonnegative continuous flow on arc $(i,j)\in A$

Decision Variables:
- $f_{ij} \geq 0$ for all $(i,j)\in A$

Objective:
\[
\min \sum_{(i,j)\in A} c_{ij} f_{ij}
\]

Subject to:

1. Source supply upper bounds (for all $i\in S$):
\[
\sum_{j: (i,j)\in A} f_{ij} \leq s_i
\]

2. Customer demand constraints (for all $j\in C$):
\[
\sum_{i: (i,j)\in A} f_{ij} \geq d_j
\]

3. Hub flow-balance constraints (for all $h\in H$):
\[
\sum_{i: (i,h)\in A} f_{ih} = \sum_{j: (h,j)\in A} f_{hj}
\]

4. Hub throughput capacity constraints (for all $h\in H$):
\[
\sum_{i: (i,h)\in A} f_{ih} \leq u_h
\]

5. Nonnegativity (for all $(i,j)\in A$):
\[
f_{ij} \geq 0
\]

Data Mapping

- $S$: all Node with NodeType = SourceSupply in file_0_view_0
- $C$: all Node with NodeType = CustomerDemand in file_0_view_0
- $H$: all Hub in file_1_view_0
- $A$: all (From, To) pairs in file_2_view_0
- $s_i$: Amount for Node $i$ in file_0_view_0, NodeType = SourceSupply
- $d_j$: Amount for Node $j$ in file_0_view_0, NodeType = CustomerDemand
- $u_h$: ThroughputCapacity for Hub $h$ in file_1_view_0
- $c_{ij}$: Cost for arc (From = $i$, To = $j$) in file_2_view_0
- $f_{ij}$: decision variable for arc (From = $i$, To = $j$) in file_2_view_0

All sets, parameters, and constraints are defined directly from the current CSV data, preserving all identifiers and bounds.