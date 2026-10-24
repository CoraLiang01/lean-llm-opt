**Mathematical Model**

Let  

- \(I\) be the set of display areas indexed by `DisplayID` from *capacity.csv*.  
- \(J\) be the set of boat types indexed by `ProductName` from *products.csv*.  

Parameters  

- \(c_i\) – capacity of display area \(i\) (units of space).  
- \(v_j\) – value of one unit of boat type \(j\).  
- \(w_j\) – weight (size) of one unit of boat type \(j\).  

Decision variables  

- \(x_{ij}\in \mathbb{Z}_{\ge 0}\) – number of units of boat type \(j\) placed in display area \(i\).

---

\[
\begin{aligned}
\max\;& \sum_{i\in I}\sum_{j\in J} v_j\,x_{ij} \\[4pt]
\text{s.t.}\;& \sum_{j\in J} w_j\,x_{ij}\;\le\; c_i , && \forall i\in I \\[4pt]
& x_{ij}\;\in\;\mathbb{Z}_{\ge 0}, && \forall i\in I,\; j\in J
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source File | Column Name |
|-----------|-------------|-------------|
| \(c_i\) | `Test_Dataset/Large-scale-or/RA_testing/RA21/BoatSales1/capacity.csv` | `Capacity` |
| \(v_j\) | `Test_Dataset/Large-scale-or/RA_testing/RA21/BoatSales1/products.csv` | `Value` |
| \(w_j\) | `Test_Dataset/Large-scale-or/RA_testing/RA21/BoatSales1/products.csv` | `Weight` |

The index sets \(I\) and \(J\) are derived from the unique values of `DisplayID` and `ProductName` in the respective files. No additional constraints or parameters are introduced beyond those specified.