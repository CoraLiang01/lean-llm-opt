## Minimum Spanning Tree Network-Design Model

**Sets**
- $V$: set of nodes (substations), from `file_0_view_0` (network_nodes.csv), $V = \{\text{N1}, \text{N2}, \text{N3}, \text{N4}, \text{N5}, \text{N6}, \text{N7}, \text{N8}\}$
- $E$: set of undirected candidate links, from `file_1_view_0` (network_edges.csv), $E = \{(i,j): \text{Node1}=i, \text{Node2}=j\}$
- For each $(i,j)\in E$, define both directions $(i,j)$ and $(j,i)$ for flow variables.

**Parameters**
- $c_{ij}$: construction cost for link $(i,j)\in E$, from `ConstructionCost` in `file_1_view_0`
- $r$: root node, from `Value` in `file_2_view_0`, $r = \text{N1}$

**Variables**
- $y_{ij} \in \{0,1\}$: 1 if undirected link $(i,j)\in E$ is built, 0 otherwise
- $f_{ij} \geq 0$: flow sent from node $i$ to node $j$ along directed arc $(i,j)$, for all $(i,j)$ and $(j,i)$ corresponding to each undirected edge in $E$

**Objective**
\[
\min \sum_{(i,j)\in E} c_{ij} \, y_{ij}
\]

**Constraints**

1. **Link count (spanning tree):**
   \[
   \sum_{(i,j)\in E} y_{ij} = |V| - 1
   \]
   (Here, $|V|=8$.)

2. **Flow conservation (connectivity):**
   For all $v \in V$:
   - If $v = r$ (root node):
     \[
     \sum_{j: (r,j)\in E \text{ or } (j,r)\in E} f_{rj} - \sum_{j: (j,r)\in E \text{ or } (r,j)\in E} f_{jr} = |V| - 1
     \]
   - If $v \neq r$:
     \[
     \sum_{j: (v,j)\in E \text{ or } (j,v)\in E} f_{vj} - \sum_{j: (j,v)\in E \text{ or } (v,j)\in E} f_{jv} = -1
     \]

3. **Flow-linking (only on built links):**
   For all $(i,j)\in E$:
   \[
   f_{ij} \leq (|V| - 1) \, y_{ij}
   \]
   \[
   f_{ji} \leq (|V| - 1) \, y_{ij}
   \]
   (Both directions for each undirected link.)

4. **Variable domains:**
   \[
   y_{ij} \in \{0,1\} \quad \forall (i,j)\in E
   \]
   \[
   f_{ij} \geq 0 \quad \forall (i,j) \text{ and } (j,i) \text{ corresponding to } (i,j)\in E
   \]

---

### Data Mapping

- **Nodes ($V$):** All `Node` entries in `file_0_view_0` (network_nodes.csv)
- **Links ($E$):** All pairs $(\text{Node1}, \text{Node2})$ in `file_1_view_0` (network_edges.csv)
- **Construction costs ($c_{ij}$):** `ConstructionCost` for each $(\text{Node1}, \text{Node2})$ in `file_1_view_0`
- **Root node ($r$):** `Value` where `Parameter` = "RootNode" in `file_2_view_0` (network_parameters.csv)
- **Variables:** $y_{ij}$ for each $(i,j)\in E$; $f_{ij}$ and $f_{ji}$ for each $(i,j)\in E$

---

**Summary:**  
This model selects $|V|-1$ links to minimize total construction cost, ensures all nodes are connected via a single spanning tree rooted at $r$, and uses directed auxiliary flows to enforce connectivity. Each link can carry flow only if it is built. All variable domains and constraints are as specified.