**Symbolic Model (TSP with 15 customers, start & end at node 1)**  

```markdown
**Index set**  
\(N = \{1,2,\dots ,15\}\)

**Parameters**  
\(d_{ij}\) – distance from node \(i\) to node \(j\) (symmetric, \(d_{ij}=d_{ji}\))

**Decision variables**  
\(x_{ij}\in\{0,1\}\) for all \(i,j\in N,\; i\neq j\) (1 if the tour goes directly from \(i\) to \(j\))  
\(u_i\in\{2,\dots ,15\}\) for all \(i\in N,\; i\neq 1\) (ordinal position of node \(i\) in the tour)

**Objective**  
\[
\min \;\sum_{i\in N}\;\sum_{\substack{j\in N\\ j\neq i}} d_{ij}\,x_{ij}
\]

**Constraints**

1. **Out‑degree** – each node has exactly one outgoing edge  
   \[
   \sum_{\substack{j\in N\\ j\neq i}} x_{ij}=1 \qquad \forall\,i\in N
   \]

2. **In‑degree** – each node has exactly one incoming edge  
   \[
   \sum_{\substack{i\in N\\ i\neq j}} x_{ij}=1 \qquad \forall\,j\in N
   \]

3. **MTZ subtour elimination** – for all \(i,j\in\{2,\dots ,15\},\, i\neq j\)  
   \[
   u_i - u_j + 14\,x_{ij}\;\le\;13
   \]

4. **Bounds on \(u\)** – positions are between 2 and 15  
   \[
   2 \;\le\; u_i \;\le\; 15 \qquad \forall\,i\in\{2,\dots ,15\}
   \]

---

### Data Mapping

| Parameter | Source | Mapping Rule |
|-----------|--------|--------------|
| \(d_{ij}\) | `file_0_view_0` (table_id `file_0_view_0`) | For each pair \((i,j)\) with \(i\neq j\):<br>• If the cell at **row \(i-1\)**, **column \(j\)** contains a numeric value, use that value.<br>• Otherwise (empty string), use the symmetric entry at **row \(j-1\)**, **column \(i\)**. |

*All other parameters are derived from the problem statement and do not require external data.*