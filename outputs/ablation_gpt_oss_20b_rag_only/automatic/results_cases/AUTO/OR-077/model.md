**Mathematical Model**

Let  

* \(I\) – set of all items in *value.csv* (indices are the values of the column **item**).  
* \(v_i\) – value of item \(i\).  
* \(w_i\) – weight of item \(i\).  
* \(x_i \in \{0,1\}\) – decision variable: 1 if item \(i\) is displayed, 0 otherwise.  
* \(W = 15\) – total weight capacity of the merchandise counter.

\[
\begin{aligned}
\max \quad & \sum_{i\in I} v_i\,x_i \\[4pt]
\text{s.t.}\quad & \sum_{i\in I} w_i\,x_i \;\le\; W \\[4pt]
& x_i \in \{0,1\}\quad \forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Table ID | Column | Parameter / Variable | Mapping Rule |
|----------|--------|----------------------|--------------|
| file_0_view_0 | item | index set \(I\) | Each distinct value of **item** becomes an element of \(I\). |
| file_0_view_0 | value | \(v_i\) | For every row, \(v_{\text{item}} = \text{value}\). |
| file_0_view_0 | weight | \(w_i\) | For every row, \(w_{\text{item}} = \text{weight}\). |
| – | – | \(W\) | Constant value 15 (given in the query). |

No filters are applied; all 140 rows are included in the index set. The model uses the data exactly as provided, with binary decision variables and a single weight‑capacity constraint.