**Mathematical Model**

Let  

* \(I\) – set of product identifiers (one per row of *RetailSalesDataset.csv*).  
* For each \(i\in I\)  
  * \(r_i\) – revenue per unit (column **Revenue**).  
  * \(d_i\) – demand quantity (column **Demand**).  
  * \(s_i\) – available inventory (column **Initial Inventory**).  

Decision variable  

\[
x_i \;\ge\; 0 \qquad \forall i\in I
\]

*\(x_i\)* = quantity of product \(i\) to be sold (fulfilled).

**Objective**

\[
\max \;\sum_{i\in I} r_i\,x_i
\]

**Constraints**

\[
\begin{aligned}
&x_i \;\le\; d_i &&\forall i\in I \quad &&\text{(do not exceed demand)}\\[4pt]
&x_i \;\le\; s_i &&\forall i\in I \quad &&\text{(do not exceed inventory)}\\[4pt]
&x_i \;\ge\; 0 &&\forall i\in I
\end{aligned}
\]

---

**Data Mapping Rules**

| Parameter | Source Table | Table ID | Column Name | Notes |
|-----------|--------------|----------|-------------|-------|
| \(I\) | RetailSalesDataset.csv | `file_0_view_0` | `Product Name` | Each distinct value becomes an index. |
| \(r_i\) | RetailSalesDataset.csv | `file_0_view_0` | `Revenue` | Revenue per unit for product \(i\). |
| \(d_i\) | RetailSalesDataset.csv | `file_0_view_0` | `Demand` | Maximum quantity that can be sold. |
| \(s_i\) | RetailSalesDataset.csv | `file_0_view_0` | `Initial Inventory` | Stock available for product \(i\). |

No additional parameters or constraints are required. The model uses only the data supplied in the observation.