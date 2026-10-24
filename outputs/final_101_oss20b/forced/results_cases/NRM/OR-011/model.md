**Mathematical Model**

```markdown
**Sets**  
\(I\) – set of products whose id_number starts with “id999”.

**Parameters**  
\(\text{Revenue}_i\) – revenue per unit of product \(i\).  
\(\text{Demand}_i\) – total deterministic demand for product \(i\) over the sales horizon.  
\(\text{Inv}_i\) – initial inventory available for product \(i\).

**Decision Variables**  
\(x_i \in \mathbb{Z}_{+}\) – number of units of product \(i\) to fulfill.

**Objective**  
\[
\max \;\sum_{i\in I} \text{Revenue}_i \; x_i
\]

**Constraints**  
\[
\begin{aligned}
x_i &\le \text{Demand}_i &&\forall i\in I \quad \text{(demand limit)}\\
x_i &\le \text{Inv}_i   &&\forall i\in I \quad \text{(inventory limit)}
\end{aligned}
\]
```

---

**Data Mapping Rules**

| Source | Table ID | Column | Mapped Parameter |
|--------|----------|--------|------------------|
| OnlineRetailSalesDataset.csv | `file_0_view_0` | `id_number` | index \(i\) |
| OnlineRetailSalesDataset.csv | `file_0_view_0` | `Revenue` | \(\text{Revenue}_i\) |
| OnlineRetailSalesDataset.csv | `file_0_view_0` | `Demand` | \(\text{Demand}_i\) |
| OnlineRetailSalesDataset.csv | `file_0_view_0` | `Initial Inventory` | \(\text{Inv}_i\) |

*Each row in the source table that satisfies the prefix filter “id999” generates one index \(i\) and the corresponding parameter values.*