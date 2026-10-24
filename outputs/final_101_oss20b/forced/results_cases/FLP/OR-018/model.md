##### Decision Variables  
\(x_i \;\ge 0\) – units of Baby product \(i\) to be sold, \(i\in I\)

##### Parameters  
\(r_i\) – revenue per unit of product \(i\)  
\(d_i\) – demand for product \(i\)  
\(\text{inv}_i\) – initial inventory of product \(i\)

##### Index Set  
\(I=\{\,i \mid \text{Product Name contains “Baby” in Salesdata.csv}\,\}\)

##### Objective Function  
\[
\max \;\sum_{i\in I} r_i\,x_i
\]

##### Constraints  
\[
\begin{aligned}
x_i &\le d_i &&\forall i\in I \quad\text{(cannot exceed demand)}\\
x_i &\le \text{inv}_i &&\forall i\in I \quad\text{(cannot exceed inventory)}\\
x_i &\ge 0 &&\forall i\in I
\end{aligned}
\]

---

## Data Mapping

| Parameter | Source Table | Source Column |
|-----------|--------------|---------------|
| \(r_i\)   | file_0_view_0 | Revenue |
| \(d_i\)   | file_0_view_0 | Demand |
| \(\text{inv}_i\) | file_0_view_0 | Initial Inventory |

The index set \(I\) is derived from the rows of **file_0_view_0** that satisfy the filter condition `Product Name contains "Baby"`. No other tables or columns are used.