**Mathematical Model**

Let  

* \(N=\{\text{Depot},A,B,C\}\) – set of all nodes.  
* \(V=\{A,B,C\}\) – set of customer nodes.  
* \(x_{ij}\in\{0,1\}\) – binary variable that equals 1 if the vehicle travels directly from node \(i\) to node \(j\).  
* \(u_i\in\mathbb{Z}\) – auxiliary position variable for customer \(i\in V\) (used for subtour elimination).

---

### Objective
\[
\min \;\sum_{i\in N}\;\sum_{\substack{j\in N\\ j\neq i}} d_{ij}\,x_{ij}
\]

### Constraints

1. **Out‑degree (each node leaves exactly one edge)**  
   \[
   \sum_{\substack{j\in N\\ j\neq i}} x_{ij}=1,\qquad \forall i\in N
   \]

2. **In‑degree (each node is entered exactly once)**  
   \[
   \sum_{\substack{i\in N\\ i\neq j}} x_{ij}=1,\qquad \forall j\in N
   \]

3. **Subtour elimination (Miller–Tucker–Zemlin)**  
   \[
   u_i - u_j + |V|\,x_{ij}\le |V|-1,\qquad \forall i,j\in V,\; i\neq j
   \]

4. **Bounds on auxiliary variables**  
   \[
   1 \le u_i \le |V|,\qquad \forall i\in V
   \]

5. **Variable domains**  
   \[
   x_{ij}\in\{0,1\},\qquad \forall i,j\in N,\; i\neq j
   \]

---

## Data Mapping

| Parameter | Source | Mapping Rule |
|-----------|--------|--------------|
| \(d_{ij}\) | `file_0_view_0` | For each pair \((i,j)\) with \(i,j\in N\), take the value in the row whose `Unnamed: 0` equals \(i\) and the column named \(j\). |
| \(N\) | `file_0_view_0` | Set of distinct values appearing in the `Unnamed: 0` column and the column headers `Depot`, `A`, `B`, `C`. |
| \(V\) | Derived | \(V = N \setminus \{\text{Depot}\}\). |

No additional data tables are required. The model uses only the distances supplied in `DistanceMatrix.csv`.