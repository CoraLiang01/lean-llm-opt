##### Index Set  
\(I=\{i \mid \text{Product Name of } i \text{ contains the string } \text{``Baby''}\}\)

##### Parameters  
\(w_i\) – revenue per unit of product \(i\)  
\(d_i\) – deterministic demand for product \(i\) over the sales horizon  
\(c_i\) – initial inventory of product \(i\)

##### Decision Variables  
\(x_i \in \mathbb{Z},\; x_i \ge 0,\; \forall i \in I\)

##### Objective Function  
\[
\max \sum_{i\in I} w_i \, x_i
\]

##### Constraints  

1. **Demand Constraints**  
\[
x_i \le d_i,\qquad \forall i \in I
\]

2. **Inventory Constraints**  
\[
x_i \le c_i,\qquad \forall i \in I
\]

##### Data Mapping  

- **Table**: `file_0_view_0`  
  - **Column**: `Product Name` → product identifier (used to define set \(I\))  
  - **Column**: `Revenue` → \(w_i\)  
  - **Column**: `Demand` → \(d_i\)  
  - **Column**: `Initial Inventory` → \(c_i\)  

The filter condition for set \(I\) is that `Product Name` contains the substring `"Baby"`.