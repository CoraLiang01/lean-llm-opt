##### Sets
- \(I\): set of all products.

##### Parameters
- \(l_i\): labor required per unit of product \(i\) \((\text{units of labor})\).
- \(m_i\): material required per unit of product \(i\) \((\text{units of material})\).
- \(p_i\): selling price per unit of product \(i\) \((\$)\).
- \(c_i\): variable cost per unit of product \(i\) \((\$)\).

##### Variables
- \(x_i\): production quantity of product \(i\) \((\text{units})\), \(x_i \in \mathbb{R}_+\).

##### Objective
\[
\max \; Z \;=\; \sum_{i\in I} (p_i - c_i)\,x_i \;-\; 4500
\]

##### Constraints
1. **Labor capacity**  
   \[
   \sum_{i\in I} l_i\,x_i \;\le\; 1650
   \]
2. **Material capacity**  
   \[
   \sum_{i\in I} m_i\,x_i \;\le\; 1850
   \]

##### Data Mapping
- **Table ID**: `file_0_view_0`
- **Columns**: `"Product Name"`, `"Labor per unit"`, `"Material per unit"`, `"Selling Price"`, `"Variable Cost"`