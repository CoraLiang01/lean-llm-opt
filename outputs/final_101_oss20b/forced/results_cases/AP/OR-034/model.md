##### Model

Let  

- \(I\) be the set of all products.  
- For each \(i\in I\)  
  - \(r_i\)  – revenue per unit (from column **Revenue**)  
  - \(d_i\)  – demand (from column **Demand**)  
  - \(s_i\)  – initial inventory (from column **Initial Inventory**)  

Decision variable  

- \(x_i\) – quantity of product \(i\) to sell (continuous, \(x_i\ge 0\))  

**Objective**

\[
\max \;\sum_{i\in I} r_i\,x_i
\]

**Constraints**

\[
\begin{aligned}
x_i &\le d_i &&\forall i\in I \quad &&\text{(cannot exceed demand)}\\
x_i &\le s_i &&\forall i\in I \quad &&\text{(cannot exceed inventory)}\\
x_i &\ge 0   &&\forall i\in I
\end{aligned}
\]

---

##### Data Mapping

| Parameter | Source Table | Source Column |
|-----------|--------------|---------------|
| \(r_i\)   | `file_0_view_0` | `Revenue` |
| \(d_i\)   | `file_0_view_0` | `Demand` |
| \(s_i\)   | `file_0_view_0` | `Initial Inventory` |
| Index set \(I\) | `file_0_view_0` | `Product Name` |

All values are taken directly from the CSV; no enumeration of individual rows is shown.