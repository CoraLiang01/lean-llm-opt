##### Index Set  
\(I=\{i \mid \text{SKU of product }i \text{ starts with } \text{ZZ}\}\)

##### Parameters  
\(w_i=\) revenue of product \(i\) (from column **Revenue**)  
\(d_i=\) demand of product \(i\) (from column **Demand**)  
\(c_i=\) initial inventory of product \(i\) (from column **Initial Inventory**)

##### Decision Variables  
\(x_i \in \mathbb{Z},\; x_i \ge 1,\; \forall i\in I\)

##### Objective Function  
\(\displaystyle \max \sum_{i\in I} w_i\,x_i\)

##### Constraints  

1. **Inventory Constraints**  
\(\displaystyle x_i \le c_i,\;\forall i\in I\)

2. **Demand Constraints**  
\(\displaystyle x_i \le d_i,\;\forall i\in I\)

##### Data Mapping  
- **Table ID**: `file_0_view_0`  
- **Columns**: `SKU`, `Revenue`, `Demand`, `Initial Inventory`