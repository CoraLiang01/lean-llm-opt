##### Index Sets  
\[
I \;=\; \{\, i \mid \text{product } i \text{ appears in } \text{MobileSalesDataset.csv}\,\}
\]

##### Parameters  
\[
\begin{aligned}
w_i &\;=\; \text{Revenue of product } i, && \forall i \in I \\
a_i &\;=\; \text{Demand of product } i, && \forall i \in I \\
c_i &\;=\; \text{Initial Inventory of product } i, && \forall i \in I
\end{aligned}
\]

##### Decision Variables  
\[
x_i \;\in\; \mathbb{Z}_{\ge 0}, \qquad \forall i \in I
\]

##### Objective Function  
\[
\max \;\sum_{i \in I} w_i \, x_i
\]

##### Constraints  

1. **Inventory Constraints**  
\[
x_i \;\le\; c_i, \qquad \forall i \in I
\]

2. **Demand Constraints**  
\[
x_i \;\le\; a_i, \qquad \forall i \in I
\]

##### Data Mapping  

| Parameter | Source Table | Source Column |
|-----------|--------------|---------------|
| \(w_i\)   | MobileSalesDataset.csv | Revenue |
| \(a_i\)   | MobileSalesDataset.csv | Demand |
| \(c_i\)   | MobileSalesDataset.csv | Initial Inventory |

All product identifiers \(i\) are taken directly from the “Product Name” column of the same table. No additional constraints or data are introduced beyond those specified in the problem description.