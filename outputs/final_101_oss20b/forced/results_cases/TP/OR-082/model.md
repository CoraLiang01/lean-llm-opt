**Mathematical Model**

Let  

* \(N=\{Depot, A, B, C\}\) – set of all locations (depot + customers).  
* \(x_{ij}\in\{0,1\}\) – binary variable that equals 1 if the van travels directly from node \(i\) to node \(j\).  
* \(u_i\) – continuous “order” variable for node \(i\in N\setminus\{Depot\}\) (used for subtour elimination).  

Parameters  

* \(d_{ij}\ge 0\) – road distance (km) from node \(i\) to node \(j\).  

The objective is to minimise the total travel distance:

\[
\min \;\sum_{i\in N}\sum_{j\in N} d_{ij}\,x_{ij}
\]

Subject to  

1. **Out‑degree** – each node sends exactly one arc:

\[
\sum_{j\in N} x_{ij}=1 \quad \forall i\in N
\]

2. **In‑degree** – each node receives exactly one arc:

\[
\sum_{i\in N} x_{ij}=1 \quad \forall j\in N
\]

3. **Subtour elimination (MTZ)** – for all distinct customer nodes \(i,j\in N\setminus\{Depot\}\):

\[
u_i - u_j + 3\,x_{ij}\;\le\;2
\]

4. **Order variable bounds** – for all customer nodes \(i\in N\setminus\{Depot\}\):

\[
1 \;\le\; u_i \;\le\; 3
\]

5. **Binary variables** – for all \(i,j\in N\):

\[
x_{ij}\in\{0,1\}
\]

---

**Data Mapping Rules**

| Parameter / Variable | Source | Mapping |
|----------------------|--------|---------|
| \(N\) | `file_0_view_0` | Set of column names excluding `"Unnamed: 0"`: `{Depot, A, B, C}` |
| \(d_{ij}\) | `file_0_view_0` | For each row `data_row_number` and column `col` (excluding `"Unnamed: 0"`), assign `d_{row_col} = value` (numeric). |
| \(x_{ij}\) | — | Decision variable, binary. |
| \(u_i\) | — | Decision variable, continuous, bounds as above. |

*The table `file_0_view_0` contains the full distance matrix; each entry `values` provides the numeric distance for the corresponding pair \((i,j)\). No additional data files are required.*