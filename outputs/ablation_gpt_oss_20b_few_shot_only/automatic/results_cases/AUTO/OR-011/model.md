**Mathematical Model**

\[
\begin{aligned}
\text{Maximize}\quad & \sum_{i\in I} R_i\,x_i \\[4pt]
\text{subject to}\quad
& x_i \;\le\; D_i &&\forall i\in I \quad\text{(demand)}\\
& x_i \;\le\; I_i &&\forall i\in I \quad\text{(inventory)}\\
& x_i \;\in\;\mathbb{Z}_{\ge 0} &&\forall i\in I
\end{aligned}
\]

**Index set**

\[
I=\{\,999\,\}
\]

**Parameters**

| Symbol | Meaning | Source column |
|--------|---------|---------------|
| \(R_i\) | Revenue per unit of product \(i\) | `Revenue` |
| \(D_i\) | Total demand for product \(i\) over the sales horizon | `Demand` |
| \(I_i\) | Initial inventory of product \(i\) | `Initial Inventory` |

**Decision variables**

* \(x_i\) – number of units of product \(i\) to fulfill (integer, non‑negative).

---

**Data Mapping**

| File path | Column name | Parameter mapped |
|-----------|-------------|------------------|
| `Test_Dataset/Large-scale-or/NRM_testing/NRM2/OnlineRetailSalesDataset.csv` | `Revenue` | \(R_i\) |
| `Test_Dataset/Large-scale-or/NRM_testing/NRM2/OnlineRetailSalesDataset.csv` | `Demand` | \(D_i\) |
| `Test_Dataset/Large-scale-or/NRM_testing/NRM2/OnlineRetailSalesDataset.csv` | `Initial Inventory` | \(I_i\) |

*Only the row with `id_number = id999` is used to instantiate the parameters for the single index element \(i=999\).*