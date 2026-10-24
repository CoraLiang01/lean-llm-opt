**Mathematical Model**

\[
\begin{aligned}
\text{Maximize}\quad & \sum_{i\in I} \text{Value}_i\,x_i \\[4pt]
\text{subject to}\quad
& \sum_{i\in I} \text{Weight}_i\,x_i \;\le\; \text{Capacity} \\[4pt]
& x_i \in \mathbb{Z}_{\ge 0}\qquad \forall i\in I
\end{aligned}
\]

**Index Sets**

- \(I\): set of vehicle types (identified by the `ProductName` column in *products.csv*).

**Parameters**

- \(\text{Value}_i\): benefit coefficient for vehicle type \(i\) (column `Value` in *products.csv*).
- \(\text{Weight}_i\): inventory weight for vehicle type \(i\) (column `Weight` in *products.csv*).
- \(\text{Capacity}\): total inventory capacity (column `Capacity` in *capacity.csv*).

**Variables**

- \(x_i\): number of units of vehicle type \(i\) to order daily, integer and non‑negative.

---

**Data Mapping**

| Parameter | File Path | Column |
|-----------|-----------|--------|
| `Capacity` | `Test_Dataset/Large-scale-or/RA_testing/RA2/CarSales1/capacity.csv` | `Capacity` |
| `Value_i` | `Test_Dataset/Large-scale-or/RA_testing/RA2/CarSales1/products.csv` | `Value` |
| `Weight_i` | `Test_Dataset/Large-scale-or/RA_testing/RA2/CarSales1/products.csv` | `Weight` |
| `I` | `Test_Dataset/Large-scale-or/RA_testing/RA2/CarSales1/products.csv` | `ProductName` (each row defines an element of \(I\)) |

All numerical values are taken directly from the specified CSV files; no additional data are introduced.