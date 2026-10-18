##### Sets  
\(I\) – set of warehouses (regions)  
\(J\) – set of stores (customers)

##### Parameters  
\(d_j\) – demand of store \(j\)  (from **file_0_view_0**, column **demand**)  
\(s_i\) – supply capacity of warehouse \(i\)  (from **file_1_view_0**, column **supply_capacity**)  
\(c_{i,j}\) – transportation cost per unit from warehouse \(i\) to store \(j\)  (from **file_2_view_0**, columns **D1**–**D5** with row identifier **Unnamed: 0**)

##### Decision Variables  
\(x_{i,j}\) – quantity shipped from warehouse \(i\) to store \(j\)  \(x_{i,j}\ge 0\)

##### Objective Function  
\[
\min \; Z \;=\; \sum_{i\in I}\;\sum_{j\in J}\; c_{i,j}\,x_{i,j}
\]

##### Constraints  

**Supply capacity constraints**  
\[
\sum_{j\in J} x_{i,j}\;\le\; s_i , \qquad \forall i\in I
\]

**Demand satisfaction constraints**  
\[
\sum_{i\in I} x_{i,j}\;=\; d_j , \qquad \forall j\in J
\]

##### Data Mapping  

| Source | Table ID | Column(s) | Mapping |
|--------|----------|-----------|---------|
| Customer demand | **file_0_view_0** | **customer** → \(J\) <br> **demand** → \(d_j\) | |
| Warehouse supply | **file_1_view_0** | **region** → \(I\) <br> **supply_capacity** → \(s_i\) | |
| Transportation costs | **file_2_view_0** | **Unnamed: 0** → \(I\) <br> **D1**–**D5** → \(c_{i,j}\) | |

This abstract model captures the GreenMart transportation problem as described.