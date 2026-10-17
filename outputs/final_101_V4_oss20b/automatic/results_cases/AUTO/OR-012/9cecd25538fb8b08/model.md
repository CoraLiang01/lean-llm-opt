**Mathematical Model**

Let  

- \(I\) be the set of all products in the dataset.  
- For each \(i\in I\)  
  - \(r_i\)  – revenue per unit (from column **Revenue**)  
  - \(d_i\)  – deterministic demand over the horizon (from column **Demand**)  
  - \(s_i\)  – initial inventory available (from column **Initial Inventory**)  

Decision variable  

- \(x_i \in \mathbb{Z}_{+}\) – number of units of product \(i\) the retailer chooses to sell.

\[
\begin{aligned}
\max \quad & \sum_{i\in I} r_i\,x_i \\[4pt]
\text{s.t.}\quad
& x_i \;\le\; d_i, \qquad &&\forall i\in I \quad \text{(demand limit)}\\
& x_i \;\le\; s_i, \qquad &&\forall i\in I \quad \text{(inventory limit)}\\
& x_i \;\ge\; 0, \qquad &&\forall i\in I \quad \text{(non‑negativity)}\\
& x_i \in \mathbb{Z}, \qquad &&\forall i\in I \quad \text{(integer units)}
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source Table | Source Column |
|-----------|--------------|---------------|
| \(I\) (index set) | `file_0_view_0` | `Product Name` |
| \(r_i\) | `file_0_view_0` | `Revenue` |
| \(d_i\) | `file_0_view_0` | `Demand` |
| \(s_i\) | `file_0_view_0` | `Initial Inventory` |

No additional constraints or parameters are required beyond those listed.