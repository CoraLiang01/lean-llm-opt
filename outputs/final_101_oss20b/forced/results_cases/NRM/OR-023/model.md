**Mathematical Model**

Let  

- \(I\) be the set of all products whose *Product_Reference* starts with **ELE‑S**.  
- For each \(i\in I\)  
  - \(R_i\) : revenue per unit (from column **Revenue**)  
  - \(D_i\) : deterministic demand over the horizon (from column **Demand**)  
  - \(I_i\) : initial inventory available (from column **Initial Inventory**)  

Decision variable  

- \(x_i \in \mathbb{Z}_{+}\) : number of units of product \(i\) to sell.

\[
\begin{aligned}
\max\;& \sum_{i\in I} R_i\,x_i \\[4pt]
\text{s.t. } & x_i \;\le\; D_i &&\forall i\in I \\[2pt]
             & x_i \;\le\; I_i &&\forall i\in I \\[2pt]
             & x_i \;\ge\; 0 &&\forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source Table | Source Column |
|-----------|--------------|---------------|
| \(R_i\)   | `file_0_view_0` | `Revenue` |
| \(D_i\)   | `file_0_view_0` | `Demand` |
| \(I_i\)   | `file_0_view_0` | `Initial Inventory` |
| Index \(i\) | `file_0_view_0` | `Product_Reference` (prefix **ELE‑S**) |

No additional data or constraints are required beyond those specified.