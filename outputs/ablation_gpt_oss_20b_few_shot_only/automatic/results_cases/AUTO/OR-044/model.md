**Mathematical Model**

\[
\begin{aligned}
\text{Maximize}\quad & \sum_{i\in I}\sum_{j\in J} V_j\,x_{ij} \\[4pt]
\text{subject to}\quad & \sum_{j\in J} W_j\,x_{ij}\;\le\; C_i , && \forall i\in I \\[4pt]
& x_{ij}\;\in\;\mathbb{Z}_{\ge 0}, && \forall i\in I,\; j\in J
\end{aligned}
\]

**Index Sets**

- \(I=\{1,\dots,8\}\) (section identifiers)  
- \(J=\{1,\dots,10\}\) (product identifiers)

**Parameters**

- \(C_i\) capacity of section \(i\) (from *capacity.csv*, column **Capacity**)  
- \(V_j\) value (revenue) of product \(j\) (from *products.csv*, column **Value**)  
- \(W_j\) space requirement of product \(j\) (from *products.csv*, column **Weight**)

**Variables**

- \(x_{ij}\) integer number of units of product \(j\) placed in section \(i\)

---

### Data Mapping

| Parameter | File Path | Column Name |
|-----------|-----------|-------------|
| \(C_i\) | `Test_Dataset/Large-scale-or/RA_testing/RA10/SalesOfASupermarket1/capacity.csv` | `Capacity` |
| \(V_j\) | `Test_Dataset/Large-scale-or/RA_testing/RA10/SalesOfASupermarket1/products.csv` | `Value` |
| \(W_j\) | `Test_Dataset/Large-scale-or/RA_testing/RA10/SalesOfASupermarket1/products.csv` | `Weight` |

*Indices \(I\) and \(J\) are derived from the `SectionID` and `ProductName` columns of the respective files.*