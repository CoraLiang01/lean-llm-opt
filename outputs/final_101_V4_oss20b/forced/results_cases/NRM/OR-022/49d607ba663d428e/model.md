**Mathematical Model**

\[
\begin{aligned}
\text{Maximize}\quad & \sum_{i\in I} r_i\,x_i \\[4pt]
\text{subject to}\quad
& x_i \;\le\; d_i &&\forall i\in I \quad\text{(demand)}\\
& x_i \;\le\; I_i &&\forall i\in I \quad\text{(inventory)}\\
& x_i \in \mathbb{Z}_+ &&\forall i\in I
\end{aligned}
\]

**Index set**

* \(I\) – set of all products classified under “27in” (identified by the values in the *Product Name* column).

**Parameters**

* \(r_i\) – revenue per unit of product \(i\) (from the *Revenue* column).  
* \(d_i\) – total demand for product \(i\) (from the *Demand* column).  
* \(I_i\) – initial inventory of product \(i\) (from the *Initial Inventory* column).

**Variables**

* \(x_i\) – number of units of product \(i\) to be fulfilled (integer, non‑negative).

---

### Data Mapping

| Parameter | Source Table | Source Column |
|-----------|--------------|---------------|
| \(r_i\)   | `file_0_view_0` | `Revenue` |
| \(d_i\)   | `file_0_view_0` | `Demand` |
| \(I_i\)   | `file_0_view_0` | `Initial Inventory` |
| \(I\)     | `file_0_view_0` | `Product Name` (each distinct value defines an index element) |

*Table `file_0_view_0` is the view produced by filtering `Salesorders.csv` for rows where *Product Name* contains “27in”.*