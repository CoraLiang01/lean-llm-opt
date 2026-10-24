**Mathematical Model**

Let  

* \(N=\{V1,\dots ,V9\}\) – set of sites (nodes).  
* \(E=\{(i,j)\mid i<j,\; (i,j)\text{ appears in } \text{network_edges.csv}\}\) – set of undirected links.  
* \(A=\{(i,j),(j,i)\mid (i,j)\in E\}\) – set of directed arcs.  
* \(r\) – root node (value from *network_parameters.csv*).  
* \(M=|N|-1=8\).

Parameters  

* \(c_{ij}\) – construction cost of undirected edge \((i,j)\) (from *network_edges.csv*).  
* \(y_{ij}\) – binary variable for edge \((i,j)\).  
* \(f_{ij}\) – non‑negative flow on directed arc \((i,j)\).

---

### Objective
\[
\min \;\sum_{(i,j)\in E} c_{ij}\,y_{ij}
\]

### Constraints  

1. **Edge count (tree size)**
   \[
   \sum_{(i,j)\in E} y_{ij}=|N|-1=8
   \]

2. **Flow conservation**

   *Root node*  
   \[
   \sum_{(r,j)\in A} f_{rj}-\sum_{(i,r)\in A} f_{ir}=|N|-1=8
   \]

   *Non‑root nodes*  
   \[
   \sum_{(i,k)\in A} f_{ik}-\sum_{(k,j)\in A} f_{kj}=1
   \quad\forall k\in N\setminus\{r\}
   \]

3. **Flow–edge linking**  
   \[
   f_{ij}\le M\,y_{ij}\qquad\forall (i,j)\in A
   \]
   (here \(y_{ij}\) refers to the undirected edge \(\{i,j\}\); \(y_{ij}=y_{ji}\)).

4. **Non‑negativity**  
   \[
   f_{ij}\ge 0\qquad\forall (i,j)\in A
   \]

5. **Binary variables**  
   \[
   y_{ij}\in\{0,1\}\qquad\forall (i,j)\in E
   \]

---

## Data Mapping

| Parameter | Source Table | Column(s) | Notes |
|-----------|--------------|-----------|-------|
| \(N\) | `file_0_view_0` | `Node` | 9 rows (V1–V9) |
| \(E\) | `file_1_view_0` | `Node1`, `Node2` | 16 undirected edges |
| \(c_{ij}\) | `file_1_view_0` | `ConstructionCost` | Cost per edge |
| \(r\) | `file_2_view_0` | `Value` (where `Parameter`=`RootNode`) | Root node V1 |
| \(M\) | — | — | Computed as \(|N|-1=8\) |

All other data (flow variables, binary variables, and constraints) are defined purely by the model structure and the above parameters.