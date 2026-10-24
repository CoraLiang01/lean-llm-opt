**Mathematical Model**

\[
\begin{aligned}
\text{Maximize}\quad & \sum_{i\in I} \text{Value}_i \; x_i \\[4pt]
\text{subject to}\quad & \sum_{i\in I} \text{Weight}_i \; x_i \;\le\; \text{Capacity} \\[4pt]
& x_i \in \mathbb{Z}_{\ge 0} \qquad \forall i\in I
\end{aligned}
\]

**Index Sets**

- \(I\): set of all products listed in *products.csv*.

**Parameters**

- \(\text{Value}_i\): benefit per unit of product \(i\).
- \(\text{Weight}_i\): weight per unit of product \(i\).
- \(\text{Capacity}\): overall stock capacity.

**Variables**

- \(x_i\): number of units of product \(i\) to order each day.

---

### Data Mapping

| Parameter | Source File | Column |
|-----------|-------------|--------|
| \(\text{Value}_i\) | `Test_Dataset/Large-scale-or/RA_testing/RA9/PharmaSalesData2/products.csv` | `Value` |
| \(\text{Weight}_i\) | `Test_Dataset/Large-scale-or/RA_testing/RA9/PharmaSalesData2/products.csv` | `Weight` |
| \(\text{Capacity}\) | `Test_Dataset/Large-scale-or/RA_testing/RA9/PharmaSalesData2/capacity.csv` | `Capacity` |

*The index set \(I\) is implicitly defined by the rows of the products file.*