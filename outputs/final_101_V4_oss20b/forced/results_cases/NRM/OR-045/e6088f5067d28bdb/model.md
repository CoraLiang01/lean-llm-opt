**Mathematical Model**

Let  

- \(I\) be the set of produce types (the 10 rows of *products.csv*).  
- For each \(i\in I\):  
  - \(w_i\) – weight of one unit of type \(i\) (from column **Weight**).  
  - \(v_i\) – benefit (value) of one unit of type \(i\) (from column **Value**).  
- \(C\) – total inventory capacity (from column **Capacity**).  

Decision variables  

\[
x_i \in \mathbb{Z}_{\ge 0}\qquad \forall i\in I
\]

Objective  

\[
\max \;\sum_{i\in I} v_i\,x_i
\]

Capacity constraint  

\[
\sum_{i\in I} w_i\,x_i \;\le\; C
\]

No other constraints are required.

---

**Data Mapping**

| Parameter | Source Table | Column |
|-----------|--------------|--------|
| \(C\) | `file_0_view_0` | `Capacity` |
| \(w_i\) | `file_1_view_0` | `Weight` |
| \(v_i\) | `file_1_view_0` | `Value` |
| Index set \(I\) | `file_1_view_0` | `ProductName` (each distinct row) |

All parameter values are taken directly from the corresponding CSV files; no additional data or assumptions are introduced.