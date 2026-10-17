**Mathematical Model**

Let  

- \(I\) be the set of product types indexed by the values of the column **ProductName** in table **file_1_view_0**.  
- \(x_i\) be the number of vehicles of type \(i\) ordered per day.  

Parameters  

- \(p_i\) – profit per vehicle of type \(i\) (column **Value** in **file_1_view_0**).  
- \(w_i\) – weight (space requirement) per vehicle of type \(i\) (column **Weight** in **file_1_view_0**).  
- \(C\) – total inventory capacity (column **Capacity** in **file_0_view_0**).  

\[
\begin{aligned}
\max\;& \sum_{i\in I} p_i\,x_i \\[4pt]
\text{s.t.}\;& \sum_{i\in I} w_i\,x_i \;\le\; C \\[4pt]
& x_i \in \mathbb{Z}_{\ge 0}\quad \forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Symbol | Source Table | Column |
|--------|---------------|--------|
| \(I\) | file_1_view_0 | ProductName |
| \(p_i\) | file_1_view_0 | Value |
| \(w_i\) | file_1_view_0 | Weight |
| \(C\) | file_0_view_0 | Capacity |

All parameters are taken directly from the specified tables; no additional data are introduced.