**Symbolic Model**

Let  

- \(I\) be the set of all products whose *Product Name* starts with “Aalop”.  
- For each \(i\in I\)  
  - \(r_i\)  – revenue per unit (from column **Revenue**)  
  - \(d_i\)  – total demand over the sales horizon (from column **Demand**)  
  - \(I_i\) – initial inventory available (from column **Initial Inventory**)  

Decision variable  

- \(x_i\in \mathbb{Z}_{+}\) – number of units of product \(i\) the restaurant intends to sell.

\[
\begin{aligned}
\max\;& \sum_{i\in I} r_i\,x_i \\[4pt]
\text{s.t. } & x_i \le d_i &&\forall i\in I \quad\text{(demand limit)}\\
             & x_i \le I_i &&\forall i\in I \quad\text{(inventory limit)}\\
             & x_i \in \mathbb{Z}_{+} &&\forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source | Table ID | Column |
|-----------|--------|----------|--------|
| \(r_i\)   | revenue per unit | `file_0_view_0` | `Revenue` |
| \(d_i\)   | demand over horizon | `file_0_view_0` | `Demand` |
| \(I_i\)   | initial inventory | `file_0_view_0` | `Initial Inventory` |
| \(I\)     | product identifiers | `file_0_view_0` | `Product Name` (filtered by prefix “Aalop”) |

No additional data or constraints are required beyond those specified in the query.