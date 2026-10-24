**Mathematical Model**

Let  

- \(I\) be the set of all products whose *Product_Name* starts with the prefix **“Books”** (as filtered in the observation).  
- For each \(i\in I\)  
  - \(r_i\)  – revenue per unit (from column **Revenue**)  
  - \(d_i\)  – deterministic demand over the horizon (from column **Demand**)  
  - \(I_i\) – initial inventory available (from column **Initial Inventory**)  

Decision variable  

- \(x_i \in \mathbb{Z}_{+}\) – number of units of product \(i\) to be fulfilled (positive integer).

\[
\begin{aligned}
\max\;& \sum_{i\in I} r_i\,x_i \\[4pt]
\text{s.t.}\;& x_i \;\le\; d_i &&\forall i\in I \\[2pt]
& x_i \;\le\; I_i &&\forall i\in I \\[2pt]
& x_i \;\ge\; 1 &&\forall i\in I \\[2pt]
& x_i \in \mathbb{Z} &&\forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source Table | Source Column |
|-----------|--------------|---------------|
| \(r_i\)   | `file_0_view_0` | `Revenue` |
| \(d_i\)   | `file_0_view_0` | `Demand` |
| \(I_i\)   | `file_0_view_0` | `Initial Inventory` |

The index set \(I\) is derived from the rows of `file_0_view_0` that satisfy the filter condition `Product_Name` starts with `"Books"`. No additional data or numeric values are introduced beyond those supplied in the observation.