**Mathematical Model**

Let  

- \(I\) – set of dairy products (index over all rows in *DairyGoodsSalesDataset.csv*).  

Parameters (all indexed by \(i\in I\))  

\[
\begin{aligned}
\text{Revenue}_i &\;=\; \text{Revenue column for product } i,\\
\text{Demand}_i   &\;=\; \text{Demand column for product } i,\\
\text{Inv}_i      &\;=\; \text{Initial Inventory column for product } i.
\end{aligned}
\]

Decision variables  

\[
x_i \;\in\; \mathbb{Z}_{+}\quad\forall i\in I
\]

(fulfilled units of product \(i\)).

**Objective**

\[
\max \;\sum_{i\in I} \text{Revenue}_i \; x_i
\]

**Constraints**

\[
\begin{aligned}
x_i &\le \text{Demand}_i &&\forall i\in I \quad\text{(demand limit)}\\[4pt]
x_i &\le \text{Inv}_i    &&\forall i\in I \quad\text{(inventory limit)}\\[4pt]
x_i &\ge 0               &&\forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source Table | Source Column |
|-----------|--------------|---------------|
| Revenue\(_i\) | `file_0_view_0` | `Revenue` |
| Demand\(_i\)  | `file_0_view_0` | `Demand` |
| Inv\(_i\)     | `file_0_view_0` | `Initial Inventory` |

The index set \(I\) is implicitly defined by the rows of the table `file_0_view_0`. No additional data or constraints are introduced beyond those specified in the query.