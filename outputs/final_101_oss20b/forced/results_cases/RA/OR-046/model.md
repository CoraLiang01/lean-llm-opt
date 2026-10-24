**Mathematical Model**

Let  

- \(I\) be the set of products (indexed by the values of **ProductName** in *file_1_view_0*).  
- \(x_i\) – number of units of product \(i\) ordered each day.  

Parameters  

- \(v_i\) – benefit (value) of product \(i\) (column **Value** in *file_1_view_0*).  
- \(w_i\) – weight (space requirement) of product \(i\) (column **Weight** in *file_1_view_0*).  
- \(C\) – total available capacity (column **Capacity** in *file_0_view_0*).  

\[
\begin{aligned}
\max\;& \sum_{i\in I} v_i\,x_i \\[4pt]
\text{s.t.}\;& \sum_{i\in I} w_i\,x_i \;\le\; C \\[4pt]
& x_i \in \mathbb{Z}_{\ge 0}\quad \forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source Table | Column |
|-----------|--------------|--------|
| \(I\) | *file_1_view_0* | **ProductName** |
| \(v_i\) | *file_1_view_0* | **Value** |
| \(w_i\) | *file_1_view_0* | **Weight** |
| \(C\) | *file_0_view_0* | **Capacity** |

All mappings use the exact table identifiers and column names returned by the CSVQA observation. No additional data or constraints are introduced beyond those specified.