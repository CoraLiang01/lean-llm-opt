##### Abstract Model

**Index set**  
\(I \;=\; \{\, i \mid \text{product } i \text{ is classified as a Book}\,\}\)

**Parameters**  
\(w_i \;=\; \text{Revenue of product } i\)  
\(d_i \;=\; \text{Demand of product } i\)  
\(c_i \;=\; \text{Initial Inventory of product } i\)

**Decision variables**  
\(x_i \;=\; \text{number of units of product } i \text{ to fulfill}\)

**Objective function**  
\[
\quad \max \;\sum_{i\in I} w_i \, x_i
\]

**Constraints**  

1. **Demand constraints**  
\[
\quad x_i \;\le\; d_i , \qquad \forall i\in I
\]

2. **Inventory constraints**  
\[
\quad x_i \;\le\; c_i , \qquad \forall i\in I
\]

3. **Variable domain**  
\[
\quad x_i \;\in\; \mathbb{Z}, \;\; x_i \;\ge\; 0 , \qquad \forall i\in I
\]

---

##### Data Mapping

| Parameter | Source table_id | Source column name |
|-----------|-----------------|--------------------|
| \(w_i\)   | file_0_view_0   | Revenue            |
| \(d_i\)   | file_0_view_0   | Demand             |
| \(c_i\)   | file_0_view_0   | Initial Inventory  |
| \(i\)     | file_0_view_0   | Product_Name       |

*The table `file_0_view_0` (alias of `DifferentStoreSales.csv`) contains all records where `Product_Name` includes the substring “Books”. No record values or counts are reproduced here.*