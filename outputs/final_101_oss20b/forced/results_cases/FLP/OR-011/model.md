**Mathematical Model**

Let  

- \(I\) be the set of all products whose `id_number` starts with **id999**.  
- For each \(i\in I\)  
  - \(r_i\)  – revenue per unit (from column **Revenue**)  
  - \(d_i\)  – demand during the horizon (from column **Demand**)  
  - \(v_i\)  – initial inventory available (from column **Initial Inventory**)  
- Decision variable  
  - \(x_i \in \mathbb{Z}_{+}\) – units of product \(i\) to fulfill  

The problem is

\[
\begin{aligned}
\max_{x}\;& \sum_{i\in I} r_i\,x_i \\[4pt]
\text{s.t. }& 0 \le x_i \le d_i && \forall i\in I \\[2pt]
& 0 \le x_i \le v_i && \forall i\in I \\[2pt]
& x_i \in \mathbb{Z} && \forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Parameter / Variable | Source | Table ID | Column |
|----------------------|--------|----------|--------|
| \(I\) | Index set | – | – |
| \(r_i\) | Revenue | `file_0_view_0` | `Revenue` |
| \(d_i\) | Demand | `file_0_view_0` | `Demand` |
| \(v_i\) | Initial Inventory | `file_0_view_0` | `Initial Inventory` |
| \(x_i\) | Decision variable | – | – |

*All values for the parameters are taken directly from the filtered rows of `OnlineRetailSalesDataset.csv` where `id_number` has the prefix `id999`.*