**Mathematical Model**

Let  

\[
I=\{\,i\mid \text{Product\_Reference}_i \text{ starts with “ELE‑S”}\,\}
\]

be the set of all products that belong to the *ELE‑S* class.

| Symbol | Meaning | Domain |
|--------|---------|--------|
| \(i\) | product index | \(I\) |
| \(r_i\) | revenue per unit of product \(i\) | \(\mathbb{R}_{+}\) |
| \(d_i\) | deterministic demand for product \(i\) | \(\mathbb{Z}_{+}\) |
| \(I_i\) | initial inventory of product \(i\) | \(\mathbb{Z}_{+}\) |
| \(x_i\) | number of units of product \(i\) to be sold | \(\mathbb{Z}_{+}\) (positive integer) |

---

### Objective
\[
\max \;\sum_{i\in I} r_i\,x_i
\]

### Constraints
\[
\begin{aligned}
&x_i \;\le\; d_i &&\forall i\in I &&\text{(demand limit)}\\[4pt]
&x_i \;\le\; I_i &&\forall i\in I &&\text{(inventory limit)}\\[4pt]
&x_i \;\in\; \mathbb{Z}_{+} &&\forall i\in I &&\text{(integer, positive)}
\end{aligned}
\]

---

**Data Mapping**

| Data source | Column name | Parameter mapped |
|-------------|-------------|------------------|
| `Test_Dataset/Large-scale-or/NRM_testing/NRM14/SalesStoreoverview.csv` | `Product_Reference` | Index set \(I\) (filter: strings starting with “ELE‑S”) |
| | `Revenue` | \(r_i\) |
| | `Demand` | \(d_i\) |
| | `Initial Inventory` | \(I_i\) |

No additional data or constraints are introduced beyond those specified in the query.