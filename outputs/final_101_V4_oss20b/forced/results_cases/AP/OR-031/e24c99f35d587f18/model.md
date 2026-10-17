**Mathematical Model**

Let  

- \(I\) be the set of products indexed by the values in column **Full_Product_Name**.  
- Parameters  
  \[
  \begin{aligned}
  r_i &\;=\;\text{Revenue per unit of product }i,\\
  d_i &\;=\;\text{Demand quantity of product }i,\\
  inv_i &\;=\;\text{Initial inventory of product }i.
  \end{aligned}
  \]
- Decision variable  
  \[
  x_i \;=\;\text{number of units of product }i\text{ to be fulfilled}.
  \]

**Objective**

\[
\max \;\sum_{i\in I} r_i\,x_i
\]

**Constraints**

\[
\begin{aligned}
0 &\le x_i &\forall i\in I,\\
x_i &\le d_i &\forall i\in I,\\
x_i &\le inv_i &\forall i\in I.
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source Table | Column |
|-----------|--------------|--------|
| \(r_i\)   | file_0_view_0 | Revenue |
| \(d_i\)   | file_0_view_0 | Demand |
| \(inv_i\) | file_0_view_0 | Initial Inventory |

The index set \(I\) is derived from the distinct values in the **Full_Product_Name** column of the same table. No numerical values are enumerated in the model.