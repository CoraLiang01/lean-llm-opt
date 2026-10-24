**Mathematical Model**

\[
\begin{aligned}
\text{Maximize}\quad & \sum_{i\in I} \text{Value}_i \; x_i \\[4pt]
\text{subject to}\quad & \sum_{i\in I} \text{Weight}_i \; x_i \;\le\; C \\[4pt]
& x_i \in \mathbb{Z}_{\ge 0}\qquad \forall i\in I
\end{aligned}
\]

**Data Mapping**

| Symbol | Source File | Column |
|--------|-------------|--------|
| \(I\) | `Test_Dataset/Large-scale-or/RA_testing/RA12/SupermarketSalesData2/products.csv` | `ProductName` |
| \(\text{Weight}_i\) | same file | `Weight` |
| \(\text{Value}_i\) | same file | `Value` |
| \(C\) | `Test_Dataset/Large-scale-or/RA_testing/RA12/SupermarketSalesData2/capacity.csv` | `Capacity` |