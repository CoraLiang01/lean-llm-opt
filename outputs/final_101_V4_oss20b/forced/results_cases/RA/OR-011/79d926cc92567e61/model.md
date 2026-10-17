**Mathematical Model**

\[
\begin{aligned}
\text{Maximize}\quad & \sum_{i\in I} \; \text{rev}_i \; x_i \\[4pt]
\text{subject to}\quad
& x_i \;\le\; \text{inv}_i && \forall i\in I \quad\text{(inventory limit)}\\
& x_i \;\le\; \text{dem}_i && \forall i\in I \quad\text{(demand limit)}\\
& x_i \;\in\; \mathbb{Z}_{\ge 0} && \forall i\in I
\end{aligned}
\]

---

### Index Sets
- \(I\): set of product identifiers with prefix **id999** (obtained from the `id_number` column).

### Parameters
| Symbol | Description | Data Mapping |
|--------|-------------|--------------|
| \(\text{rev}_i\) | Revenue per unit of product \(i\) | `table_id`: **file_0_view_0**, `column`: **Revenue** |
| \(\text{dem}_i\) | Demand quantity for product \(i\) | `table_id`: **file_0_view_0**, `column`: **Demand** |
| \(\text{inv}_i\) | Initial inventory for product \(i\) | `table_id`: **file_0_view_0**, `column`: **Initial Inventory** |

### Variables
- \(x_i\): number of units of product \(i\) to fulfill (integer, \(\ge 0\)).

The model uses the single record returned for `id999` as the sole element of \(I\). All constraints and the objective are defined symbolically; numerical values are supplied via the data mapping.