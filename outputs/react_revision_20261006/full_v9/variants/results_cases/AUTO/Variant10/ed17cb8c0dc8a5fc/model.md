## Mathematical Model

**Sets**
- $N$: set of nodes (substations), from `file_0_view_0`, column `Node`.
- $E$: set of undirected candidate links, from `file_1_view_0`, columns `Node1`, `Node2`.
- For each $e = (i,j) \in E$, define both directions $(i,j)$ and $(j,i)$ for flow variables.
- $r$: root node, from `file_2_view_0`, column `Value` where `Parameter` = "RootNode".

**Parameters**
- $c_{ij}$: construction cost of link $e = (i,j)$, from `file_1_view_0`, column `ConstructionCost`.
- $n = |N|$: number of nodes.

**Variables**
- $y_{ij} \in \{0,1\}$: 1 if undirected link $e = (i,j)$ is built, 0 otherwise.
- $f_{ij} \geq 0$: flow sent from node $i$ to node $j$ along directed arc $(i,j)$, for all $(i,j)$ where $(i,j) \in E$ or $(j,i) \in E$.

**Objective**
$$
\min \sum_{(i,j) \in E} c_{ij} \, y_{ij}
$$

**Constraints**

1. **Link count (spanning tree):**
$$
\sum_{(i,j) \in E} y_{ij} = n - 1
$$

2. **Flow conservation (connectivity):**
For all $k \in N \setminus \{r\}$:
$$
\sum_{j: (j,k) \in E \text{ or } (k,j) \in E} f_{jk} - \sum_{j: (k,j) \in E \text{ or } (j,k) \in E} f_{kj} = 1
$$

For the root node $r$:
$$
\sum_{j: (r,j) \in E \text{ or } (j,r) \in E} f_{rj} - \sum_{j: (j,r) \in E \text{ or } (r,j) \in E} f_{jr} = 1 - (n-1) = -(n-2)
$$

3. **Flow-link coupling:**
For all $(i,j) \in E$:
$$
f_{ij} \leq (n-1) \, y_{ij}
$$
$$
f_{ji} \leq (n-1) \, y_{ij}
$$

4. **Variable domains:**
$$
y_{ij} \in \{0,1\} \quad \forall (i,j) \in E
$$
$$
f_{ij} \geq 0 \quad \forall (i,j) \text{ such that } (i,j) \in E \text{ or } (j,i) \in E
$$

---

### Data Mapping

- **Nodes ($N$):** All values in `network_nodes.csv` (`file_0_view_0`), column `Node`.
- **Edges ($E$):** All rows in `network_edges.csv` (`file_1_view_0`), columns `Node1`, `Node2`.
- **Construction costs ($c_{ij}$):** `ConstructionCost` in `network_edges.csv` (`file_1_view_0`).
- **Root node ($r$):** Value in `network_parameters.csv` (`file_2_view_0`), column `Value` where `Parameter` = "RootNode".
- **$n$:** Number of rows in `network_nodes.csv` (`file_0_view_0`).

All indices, parameters, and variables are defined exactly as above, with all sets and coefficients mapped directly to the provided data.