**Mathematical Model**

Let  

- \(I\) be the set of items (indexed by the values in column **item** of *value.csv*).  
- \(x_i \in \{0,1\}\) be the binary decision variable indicating whether item \(i\) is selected.  

Parameters  

- \(v_i\) – value of item \(i\) (column **value**).  
- \(w_i\) – weight of item \(i\) (column **weight**).  

Capacity  

- \(W = 15\) – total weight that can be carried.

\[
\begin{aligned}
\max \quad & \sum_{i\in I} v_i\,x_i \\[4pt]
\text{s.t.}\quad & \sum_{i\in I} w_i\,x_i \;\le\; W \\[4pt]
& x_i \in \{0,1\} \quad \forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source | Table ID | Column |
|-----------|--------|----------|--------|
| \(v_i\)   | value  | file_0_view_0 | value |
| \(w_i\)   | weight | file_0_view_0 | weight |
| \(I\)     | item   | file_0_view_0 | item  |
| \(W\)     | constant | – | – |

*All indices \(i\) are taken from the distinct values in the **item** column of *value.csv*.*