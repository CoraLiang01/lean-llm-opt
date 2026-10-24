**Symbolic Model**

Let  

* \(N=\{0,1,2,3\}\) be the set of nodes, where  
  * \(0\) = Depot,  
  * \(1\) = A,  
  * \(2\) = B,  
  * \(3\) = C.  

* \(d_{ij}\) – distance (km) from node \(i\) to node \(j\) (parameter).  

* \(x_{ij}\in\{0,1\}\) – binary variable that equals 1 if the van travels directly from node \(i\) to node \(j\).  

* \(u_i\in\{1,2,3\}\) – auxiliary position variable for nodes \(i\in\{1,2,3\}\) (Depot is excluded).  

---

### Objective
\[
\min \;\sum_{i\in N}\sum_{\substack{j\in N\\ j\neq i}} d_{ij}\,x_{ij}
\]

### Constraints

1. **Out‑degree (each node leaves once)**  
   \[
   \sum_{\substack{j\in N\\ j\neq i}} x_{ij}=1 \quad \forall i\in N
   \]

2. **In‑degree (each node is entered once)**  
   \[
   \sum_{\substack{i\in N\\ i\neq j}} x_{ij}=1 \quad \forall j\in N
   \]

3. **Subtour elimination (MTZ)**  
   \[
   u_i - u_j + 4\,x_{ij}\le 3 \quad \forall i,j\in\{1,2,3\},\; i\neq j
   \]

4. **Variable domains**  
   \[
   x_{ij}\in\{0,1\}\;\forall i\neq j,\qquad
   u_i\in\{1,2,3\}\;\forall i\in\{1,2,3\}
   \]

---

**Data Mapping**

| Parameter | Source | Mapping Rule |
|-----------|--------|--------------|
| \(d_{ij}\) | `file_0_view_0` (DistanceMatrix.csv) | For each pair \(i\neq j\) in \(N\):<br>• Locate the row where `Unnamed: 0` = node \(i\).<br>• Take the value in the column named after node \(j\).<br>• Assign this value to \(d_{ij}\).<br>Nodes are identified by the strings `"Depot"`, `"A"`, `"B"`, `"C"` corresponding to indices 0,1,2,3 respectively. |

No other data are required. The model above fully specifies the traveling‑salesman problem for the single van visiting customers A, B, C exactly once and returning to the depot, minimizing total travel distance.