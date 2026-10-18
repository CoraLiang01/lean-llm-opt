##### Model

**Index Set**  
\( I = \{1,2,\dots,24\} \)  (one index for each product in the dataset)

**Parameters**  
\( w_i \) – revenue per unit for product \( i \) (from column *Revenue*)  
\( d_i \) – deterministic demand for product \( i \) (from column *Demand*)  
\( c_i \) – initial inventory for product \( i \) (from column *Initial Inventory*)

**Decision Variables**  
\( x_i \in \mathbb{Z}_{\ge 0} \) – number of units of product \( i \) to fulfill

**Objective Function**  
\[
\max \;\sum_{i \in I} w_i \, x_i
\]

**Constraints**

1. **Inventory Constraints**  
\[
x_i \le c_i , \quad \forall i \in I
\]

2. **Demand Constraints**  
\[
x_i \le d_i , \quad \forall i \in I
\]

3. **Variable Domain**  
\[
x_i \in \mathbb{Z}_{\ge 0}, \quad \forall i \in I
\]

##### Data Mapping

| Parameter | Source Table | Source Column |
|-----------|--------------|---------------|
| \( w_i \) | WomenClothingEcommerceSalesData.csv | Revenue |
| \( d_i \) | WomenClothingEcommerceSalesData.csv | Demand |
| \( c_i \) | WomenClothingEcommerceSalesData.csv | Initial Inventory |