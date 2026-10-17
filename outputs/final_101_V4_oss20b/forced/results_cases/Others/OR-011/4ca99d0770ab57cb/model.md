**Mathematical Model**

Let  

* \(I=\{i\mid \text{row }i\text{ of table }file\_0\_view\_0\text{ satisfies }id\_number\text{ prefix }'id999'\}\).

For each \(i\in I\) define

* \(x_i\in\mathbb{Z}_{\ge 0}\) – units of product \(i\) to be fulfilled.

Parameters (obtained from the filtered table):

* \(\text{Rev}_i\) – revenue per unit of product \(i\) (column **Revenue**).
* \(\text{Dem}_i\) – demand of product \(i\) during the horizon (column **Demand**).
* \(\text{Inv}_i\) – initial inventory of product \(i\) (column **Initial Inventory**).

The optimization problem is

\[
\begin{aligned}
\max\;& \sum_{i\in I}\text{Rev}_i\,x_i \\[4pt]
\text{s.t. }& x_i \;\le\; \text{Inv}_i &&\forall i\in I \quad\text{(inventory)}\\
& x_i \;\le\; \text{Dem}_i &&\forall i\in I \quad\text{(demand)}\\
& x_i \;\in\;\mathbb{Z}_{\ge 0} &&\forall i\in I .
\end{aligned}
\]

---

**Data Mapping Rules**

| Parameter | Source | Column | Filter |
|-----------|--------|--------|--------|
| \(\text{Rev}_i\) | table `file_0_view_0` | `Revenue` | row where `id_number` starts with `id999` |
| \(\text{Dem}_i\) | table `file_0_view_0` | `Demand` | same filter |
| \(\text{Inv}_i\) | table `file_0_view_0` | `Initial Inventory` | same filter |

The index set \(I\) is defined by the filter condition on `id_number`. No other rows are included. All parameters are taken directly from the corresponding columns of the filtered rows. No additional constraints or variables are introduced beyond those specified.