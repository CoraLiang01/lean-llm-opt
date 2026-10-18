##### Model

**Sets**  
\(I\) – set of potential warehouses.  
\(J\) – set of stores.

**Parameters**  
\(f_i\) – opening cost of warehouse \(i\).  
\(s_i\) – capacity of warehouse \(i\).  
\(d_j\) – demand of store \(j\).  
\(c_{ij}\) – transportation cost from warehouse \(i\) to store \(j\).

**Variables**  
\(y_i\) – binary variable, \(y_i=1\) if warehouse \(i\) is opened, \(0\) otherwise.  
\(x_{ij}\) – quantity shipped from warehouse \(i\) to store \(j\).

**Objective Function**  

$$\min \sum_{i\in I} f_i\,y_i \;+\; \sum_{i\in I}\sum_{j\in J} c_{ij}\,x_{ij}$$

**Constraints**

1. **Demand satisfaction**  

   $$\sum_{i\in I} x_{ij} \;=\; d_j \quad \forall j\in J$$

2. **Warehouse capacity**  

   $$\sum_{j\in J} x_{ij} \;\le\; s_i\,y_i \quad \forall i\in I$$

3. **Non‑negativity**  

   $$x_{ij} \;\ge\; 0 \quad \forall i\in I,\; j\in J$$

4. **Binary opening decision**  

   $$y_i \;\in\; \{0,1\} \quad \forall i\in I$$

##### Data Mapping

| Table ID | Exact column names |
|----------|--------------------|
| file_0_view_0 | `"Warehouse (i)"`, `"Opening Cost (fi)"`, `"Capacity (units)"` |
| file_1_view_0 | `"Store (j)"`, `"Demand (units, dj)"` |
| file_2_view_0 | `"Unnamed: 0"`, `"W1"`, `"W2"`, `"W3"`, `"W4"`, `"W5"`, `"W6"`, `"W7"`, `"W8"`, `"W9"`, `"W10"`, `"W11"` |

*The transportation cost parameter \(c_{ij}\) is obtained from column `Wi` in the row where `Unnamed: 0` equals the store identifier `Wj`.*