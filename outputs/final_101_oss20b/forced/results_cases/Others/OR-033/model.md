**Mathematical Model**

Let  

* \(I\) – set of Baby products that satisfy the filter  
* \(r_i\) – revenue per unit of product \(i\) (from *Revenue*)  
* \(d_i\) – demand for product \(i\) (from *Demand*)  
* \(s_i\) – initial inventory of product \(i\) (from *Initial Inventory*)  

Decision variable  

* \(x_i \in \mathbb{Z}_{+}\) – number of units of product \(i\) to be fulfilled  

\[
\begin{aligned}
\max \quad & \sum_{i\in I} r_i\,x_i \\[4pt]
\text{s.t.}\quad
& 0 \le x_i \le d_i && \forall i\in I \\[2pt]
& 0 \le x_i \le s_i && \forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Table ID | Column | Source | Filter |
|----------|--------|--------|--------|
| `file_0_view_0` | `Product Name` | `EuropeSalesRecords.csv` | `contains "Baby"` |
| `file_0_view_0` | `Revenue` | `EuropeSalesRecords.csv` | – |
| `file_0_view_0` | `Demand` | `EuropeSalesRecords.csv` | – |
| `file_0_view_0` | `Initial Inventory` | `EuropeSalesRecords.csv` | – |

*Index set* \(I\) is the set of rows returned after applying the filter on `Product Name`.  
*Parameters* \(r_i, d_i, s_i\) are taken directly from the corresponding columns of the filtered rows.  
*Decision variables* \(x_i\) are defined for each \(i \in I\).