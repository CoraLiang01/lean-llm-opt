##### Index Set  
\(I\) – set of all ‘27in’ products in the promotional event.

##### Parameters  
\(w_i\) – revenue per unit of product \(i\) (from column **Revenue**).  
\(d_i\) – total demand for product \(i\) over the sales horizon (from column **Demand**).  
\(c_i\) – initial inventory available for product \(i\) (from column **Initial Inventory**).

##### Variables  
\(x_i\) – number of units of product \(i\) to be fulfilled, \(x_i \in \mathbb{Z},\; x_i \ge 0,\; \forall i \in I\).

##### Objective Function  
\[
\quad \quad \max \quad \sum_{i \in I} w_i \cdot x_i
\]

##### Constraints  

###### 1. Inventory Constraints  
\[
\quad \quad x_i \le c_i, \quad \forall i \in I
\]

###### 2. Demand Constraints  
\[
x_i \le d_i, \quad \forall i \in I
\]

##### Data Mapping  
- **Table ID**: `file_0_view_0`  
- **Columns**:  
  - `Revenue` → parameter \(w_i\)  
  - `Demand` → parameter \(d_i\)  
  - `Initial Inventory` → parameter \(c_i\)