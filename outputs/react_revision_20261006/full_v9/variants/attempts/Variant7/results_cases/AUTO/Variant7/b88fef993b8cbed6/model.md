##### Mathematical Model

Let:
- $N$ = set of all nodes (sources, hubs, customers)
- $A$ = set of directed arcs $(i,j)$ with per-unit cost $c_{ij}$ (from arc_costs.csv)
- $S$ = set of source nodes (nodes with NodeType = SourceSupply)
- $H$ = set of hub nodes (from hub_capacity.csv)
- $C$ = set of customer nodes (nodes with NodeType = CustomerDemand)
- $f_{ij} \geq 0$ = flow on arc $(i,j) \in A$

Parameters:
- $s_i$ = supply at source $i \in S$ (from node_supply_demand.csv)
- $d_j$ = demand at customer $j \in C$ (from node_supply_demand.csv)
- $u_h$ = throughput capacity at hub $h \in H$ (from hub_capacity.csv)
- $c_{ij}$ = per-unit cost on arc $(i,j) \in A$ (from arc_costs.csv)

Objective:
$$
\min \sum_{(i,j)\in A} c_{ij} f_{ij}
$$

Subject to:

1. **Source supply upper bounds** (for all $i \in S$):
$$
\sum_{j: (i,j)\in A} f_{ij} \leq s_i
$$

2. **Customer demand constraints** (for all $j \in C$):
$$
\sum_{i: (i,j)\in A} f_{ij} \geq d_j
$$

3. **Flow balance at each hub** (for all $h \in H$):
$$
\sum_{i: (i,h)\in A} f_{ih} = \sum_{j: (h,j)\in A} f_{hj}
$$

4. **Hub throughput capacity** (for all $h \in H$):
$$
\sum_{i: (i,h)\in A} f_{ih} \leq u_h
$$

5. **Nonnegativity** (for all $(i,j)\in A$):
$$
f_{ij} \geq 0
$$

##### Data Mapping

- **Source supply and customer demand**:  
  - Table: node_supply_demand.csv (table_id: file_0_view_0)  
    - $S = \{$ rows with NodeType = "SourceSupply" $\}$, $s_i$ from Amount  
    - $C = \{$ rows with NodeType = "CustomerDemand" $\}$, $d_j$ from Amount  
    - Node names from Node column

- **Hub throughput capacity**:  
  - Table: hub_capacity.csv (table_id: file_1_view_0)  
    - $H = \{$ Hub column $\}$, $u_h$ from ThroughputCapacity

- **Arc set and costs**:  
  - Table: arc_costs.csv (table_id: file_2_view_0)  
    - $A = \{$ all rows $\}$, arcs $(i,j)$ from From, To columns  
    - $c_{ij}$ from Cost column

- **Variables**:  
  - $f_{ij}$ for each $(i,j)\in A$ (all arcs in arc_costs.csv)

All index sets, parameters, and constraints are defined directly from the above tables, preserving all identifiers and coefficients as in the source data.