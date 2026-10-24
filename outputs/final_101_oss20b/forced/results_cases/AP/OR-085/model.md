**Mathematical Model**

Let  

\[
N=\{1,2,\dots ,15\}
\]

\[
d_{ij}\;=\;\text{distance between location }i\text{ and }j\quad (i\neq j)
\]

\[
x_{ij}\;=\;\begin{cases}
1 & \text{if the tour goes directly from }i\text{ to }j\\
0 & \text{otherwise}
\end{cases}\qquad (i\neq j)
\]

\[
u_i\;=\;\text{position of node }i\text{ in the tour}\qquad (i\neq 1),\;\;u_1=1
\]

---

### Objective
\[
\min \;\sum_{\substack{i,j\in N\\ i\neq j}} d_{ij}\,x_{ij}
\]

### Constraints

1. **Out‑degree**  
   \[
   \sum_{\substack{j\in N\\ j\neq i}} x_{ij}=1 \qquad \forall\,i\in N
   \]

2. **In‑degree**  
   \[
   \sum_{\substack{i\in N\\ i\neq j}} x_{ij}=1 \qquad \forall\,j\in N
   \]

3. **MTZ Subtour Elimination**  
   \[
   u_i - u_j + 15\,x_{ij}\;\le\;14 \qquad \forall\,i,j\in N,\; i\neq j,\; i\neq 1,\; j\neq 1
   \]

4. **Variable Domains**  
   \[
   x_{ij}\in\{0,1\}\quad\forall\,i\neq j,\qquad
   u_i\in\{2,\dots ,15\}\quad\forall\,i\neq 1
   \]

---

**Data Mapping**

```json
{
  "table_id": "file_0_view_0",
  "columns": ["2","3","4","5","6","7","8","9","10","11","12","13","14","15"],
  "rows": 15,
  "parameter": "d",
  "mapping_rule": [
    {
      "type": "upper_triangle",
      "data_row_number": "i-1",
      "source_column": "j",
      "condition": "i < j"
    },
    {
      "type": "symmetry",
      "source": "d_{ji}",
      "condition": "i > j"
    }
  ]
}
```

*Explanation of the mapping rule*  
- For every pair of distinct nodes \(i<j\), the distance \(d_{ij}\) is taken from the cell in the CSV where the row index equals \(i-1\) (since the first row corresponds to node 1) and the column name equals the string representation of \(j\).  
- For \(i>j\), the distance is symmetric: \(d_{ij}=d_{ji}\).  

All other data (index set \(N\), variable domains, and constraints) are defined symbolically as shown above.