##### Index Sets  

- \(I\) : set of all products in the catalog.  

##### Parameters  

- \(w_i\) : revenue per unit of product \(i\) (obtained from the **Revenue** column).  
- \(d_i\) : deterministic demand for product \(i\) over the sales horizon (obtained from the **Demand** column).  
- \(c_i\) : initial inventory available for product \(i\) (obtained from the **Initial Inventory** column).  

##### Decision Variables  

- \(x_i \in \mathbb{Z}_{+}\) : number of units of product \(i\) to fulfill.  

##### Objective Function  

\[
\max \sum_{i \in I} w_i \, x_i
\]

##### Constraints  

1. **Demand constraints**  
\[
x_i \le d_i , \qquad \forall i \in I
\]

2. **Inventory constraints**  
\[
x_i \le c_i , \qquad \forall i \in I
\]

3. **Non‑negativity / integrality**  
\[
x_i \in \mathbb{Z}_{+}, \qquad \forall i \in I
\]

##### Data Mapping  

| Table ID | Column Name | Parameter |
|----------|-------------|-----------|
| file_0_view_0 | Revenue | \(w_i\) |
| file_0_view_0 | Demand | \(d_i\) |
| file_0_view_0 | Initial Inventory | \(c_i\) |

The model uses the data from the table **file_0_view_0** (Salesofsummerclothes.csv) and maps each column to the corresponding parameter as shown above.