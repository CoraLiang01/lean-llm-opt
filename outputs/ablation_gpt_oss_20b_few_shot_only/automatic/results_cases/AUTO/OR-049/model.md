**Mathematical Model**

\[
\begin{aligned}
\text{Maximize}\quad & Z = \sum_{i\in I}\sum_{j\in J} \text{Value}_j \; x_{ij} \\[4pt]
\text{subject to}\quad
& \sum_{j\in J} \text{Weight}_j \; x_{ij} \;\le\; \text{Capacity}_i,
   && \forall i\in I \\[4pt]
& x_{ij}\;\in\;\mathbb{Z}_{\ge 0},
   && \forall i\in I,\; j\in J
\end{aligned}
\]

**Index Sets**

| Symbol | Meaning |
|--------|---------|
| \(I\)  | Set of shelves (ShelfID) |
| \(J\)  | Set of products (ProductName) |

**Parameters**

| Symbol | Source | Column |
|--------|--------|--------|
| \(\text{Capacity}_i\) | `Test_Dataset/Large-scale-or/RA_testing/RA15/RetailSalesAnalysis1/capacity.csv` | `Capacity` |
| \(\text{Value}_j\) | `Test_Dataset/Large-scale-or/RA_testing/RA15/RetailSalesAnalysis1/products.csv` | `Value` |
| \(\text{Weight}_j\) | `Test_Dataset/Large-scale-or/RA_testing/RA15/RetailSalesAnalysis1/products.csv` | `Weight` |

**Variables**

| Symbol | Domain | Meaning |
|--------|--------|---------|
| \(x_{ij}\) | \(\mathbb{Z}_{\ge 0}\) | Number of units of product \(j\) placed on shelf \(i\) |

---