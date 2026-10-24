**Mathematical Model**

| Symbol | Meaning |
|--------|---------|
| \(I\) | Set of all products (indexed by the *Product Name* in the data file) |
| \(r_i\) | Revenue per unit of product \(i\) |
| \(d_i\) | Total deterministic demand for product \(i\) |
| \(I_i\) | Initial inventory available for product \(i\) |
| \(x_i\) | Integer number of units of product \(i\) to fulfill |

**Variables**

\[
x_i \in \mathbb{Z}_{+}\quad \forall i\in I
\]

**Objective**

\[
\max \;\; \sum_{i\in I} r_i\,x_i
\]

**Constraints**

1. **Demand limits**

\[
x_i \;\le\; d_i \qquad \forall i\in I
\]

2. **Inventory limits**

\[
x_i \;\le\; I_i \qquad \forall i\in I
\]

---

**Data Mapping**

| Parameter | Source File | Source Column |
|-----------|-------------|---------------|
| \(I\) (index set) | `Test_Dataset/Large-scale-or/NRM_testing/NRM19/WomenClothingEcommerceSalesData.csv` | *Product Name* |
| \(r_i\) | same file | *Revenue* |
| \(d_i\) | same file | *Demand* |
| \(I_i\) | same file | *Initial Inventory* |

No additional data or constraints are introduced beyond those explicitly stated in the query.