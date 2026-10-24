**Mathematical Model**

\[
\begin{aligned}
\text{Maximize}\quad & \sum_{i\in I} \text{Value}_i \; x_i \\[4pt]
\text{subject to}\quad
& \sum_{i\in I} \text{Weight}_i \; x_i \;\le\; \text{Capacity} \\[4pt]
& x_i \;\in\; \mathbb{Z}_{\ge 0}\qquad \forall i\in I
\end{aligned}
\]

**Index Sets**

- \(I\): set of all product types (identified by the values in the `ProductName` column of *products.csv*).

**Parameters**

- \(\text{Weight}_i\): weight per unit of product \(i\) (from the `Weight` column of *products.csv*).
- \(\text{Value}_i\): benefit per unit of product \(i\) (from the `Value` column of *products.csv*).
- \(\text{Capacity}\): overall weight capacity (from the `Capacity` column of *capacity.csv*).

**Variables**

- \(x_i\): integer number of units of product \(i\) to order daily.

---

### Data Mapping

| Parameter | Source File | Column Name |
|-----------|-------------|-------------|
| \(\text{Weight}_i\) | `Test_Dataset/Large-scale-or/RA_testing/RA11/SupermarketSalesData1/products.csv` | `Weight` |
| \(\text{Value}_i\) | `Test_Dataset/Large-scale-or/RA_testing/RA11/SupermarketSalesData1/products.csv` | `Value` |
| \(\text{Capacity}\) | `Test_Dataset/Large-scale-or/RA_testing/RA11/SupermarketSalesData1/capacity.csv` | `Capacity` |

*The index set \(I\) is derived from the `ProductName` column of the products file.*