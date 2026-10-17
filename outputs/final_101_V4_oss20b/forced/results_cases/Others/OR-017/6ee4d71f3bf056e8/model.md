**Mathematical Model**

Let  

* \(I\) be the set of all product identifiers whose SKU starts with the prefix **ZZ** (obtained from the filtered view `file_0_view_0`).  
* For each \(i\in I\) define the parameters  

\[
\begin{aligned}
\text{rev}_i &:= \text{Revenue of SKU }i \quad (\text{RMB per unit})\\
\text{dem}_i &:= \text{Demand of SKU }i \quad (\text{units})\\
\text{inv}_i &:= \text{Initial Inventory of SKU }i \quad (\text{units})
\end{aligned}
\]

* Decision variable  

\[
x_i \in \mathbb{Z}_{\ge 0}\qquad\text{(units of SKU }i\text{ to sell)}
\]

**Objective**

\[
\max \;\; \sum_{i\in I} \text{rev}_i \, x_i
\]

**Constraints**

\[
\begin{aligned}
x_i &\le \text{dem}_i &&\forall i\in I \quad\text{(cannot sell more than demand)}\\[4pt]
x_i &\le \text{inv}_i &&\forall i\in I \quad\text{(cannot sell more than available inventory)}\\[4pt]
x_i &\ge 0 &&\forall i\in I
\end{aligned}
\]

---

### Data Mapping

| Parameter | Source | Table ID | Column |
|-----------|--------|----------|--------|
| \(\text{rev}_i\) | Revenue | `file_0_view_0` | `Revenue` |
| \(\text{dem}_i\) | Demand | `file_0_view_0` | `Demand` |
| \(\text{inv}_i\) | Initial Inventory | `file_0_view_0` | `Initial Inventory` |
| \(I\) | SKU prefix filter | `file_0_view_0` | `SKU` (prefix `"ZZ"`) |

All parameters are taken directly from the filtered rows of `RetailStoreSalesTransactions(ScannerData).csv`. No additional data or assumptions are introduced.