**Mathematical Model**

Let  

- \(I\) – set of all dairy products (indexed by the value of *Full_Product_Name*).  

Parameters (indexed by \(i\in I\))  

\[
\begin{aligned}
\text{Revenue}_i &\;=\; \text{Revenue of product } i,\\
\text{Demand}_i   &\;=\; \text{Demand of product } i,\\
\text{Inv}_i      &\;=\; \text{Initial Inventory of product } i.
\end{aligned}
\]

Decision variables  

\[
x_i \;\in\; \mathbb{Z}_{+}\qquad \forall i\in I
\]

(fulfilled units of product \(i\)).

**Objective**

\[
\max \;\sum_{i\in I} \text{Revenue}_i \, x_i
\]

**Constraints**

\[
\begin{aligned}
x_i &\le \text{Demand}_i &&\forall i\in I \quad\text{(demand limit)}\\[4pt]
x_i &\le \text{Inv}_i    &&\forall i\in I \quad\text{(inventory limit)}\\[4pt]
x_i &\ge 0              &&\forall i\in I \quad\text{(non‑negativity)}
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source Table | Source Column |
|-----------|--------------|---------------|
| \(\text{Revenue}_i\) | `file_0_view_0` | `Revenue` |
| \(\text{Demand}_i\)  | `file_0_view_0` | `Demand` |
| \(\text{Inv}_i\)     | `file_0_view_0` | `Initial Inventory` |

The index set \(I\) is derived from the distinct values in the `Full_Product_Name` column of the same table. No additional data or constraints are introduced beyond those specified in the query.