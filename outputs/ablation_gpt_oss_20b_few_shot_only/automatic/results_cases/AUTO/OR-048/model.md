**Mathematical Model**

Let  

- \(I=\{i\mid i\text{ is a StorageID}\}\)  
- \(J=\{j\mid j\text{ is a ProductName}\}\)

Parameters  

- \(C_i\) – capacity of storage area \(i\)  
- \(V_j\) – value of product \(j\)  
- \(W_j\) – weight (size) of product \(j\)

Decision variables  

- \(x_{ij}\in \mathbb{Z}_{\ge 0}\) – number of units of product \(j\) placed in storage area \(i\)

\[
\begin{aligned}
\max\;& \sum_{i\in I}\sum_{j\in J} V_j\,x_{ij} \\[4pt]
\text{s.t.}\;& \sum_{j\in J} W_j\,x_{ij}\;\le\; C_i , && \forall i\in I \\[4pt]
& x_{ij}\;\in\;\mathbb{Z}_{\ge 0}, && \forall i\in I,\; \forall j\in J
\end{aligned}
\]

---

**Data Mapping**

| Parameter | File Path | Column Name |
|-----------|-----------|-------------|
| \(C_i\) | `Test_Dataset/Large-scale-or/RA_testing/RA14/AmazonProductsSalesDataset2023/capacity.csv` | `Capacity` |
| \(V_j\) | `Test_Dataset/Large-scale-or/RA_testing/RA14/AmazonProductsSalesDataset2023/products.csv` | `Value` |
| \(W_j\) | `Test_Dataset/Large-scale-or/RA_testing/RA14/AmazonProductsSalesDataset2023/products.csv` | `Weight` |

Indices \(I\) and \(J\) are derived from the `StorageID` and `ProductName` columns of the respective files.