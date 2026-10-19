##### Abstract Model

**Index Set**  
\( I = \{1,2,3\} \)

**Parameters**  
\( w_i \) – revenue per unit of product \( i \)  
\( a_i \) – deterministic total demand for product \( i \) over the sales horizon  
\( c_i \) – initial inventory available for product \( i \)

**Variables**  
\( x_i \in \mathbb{Z},\; x_i \ge 1 \) – number of units of product \( i \) to fulfill

**Objective Function**  
\[
\quad \quad \max \quad \sum_{i \in I} w_i \cdot x_i
\]

**Constraints**

1. **Inventory Constraints**  
\[
\quad \quad x_i \le c_i , \quad \forall i \in I
\]

2. **Demand Constraints**  
\[
x_i \le a_i , \quad \forall i \in I
\]

3. **Variable Domain**  
\[
x_i \in \mathbb{Z},\; x_i \ge 1 , \quad \forall i \in I
\]

##### Data Mapping

| Source | Table ID | Columns Used |
|--------|----------|--------------|
| Organ product data | SupermartGrocerySales-RetailAnalyticsDataset.csv | Revenue, Demand, Initial Inventory |
| Filter | Sub Category contains “Organ” |  |

The parameter values \( w_i, a_i, c_i \) are extracted from the specified columns of the CSV file, filtered to include only rows where the Sub Category contains “Organ”.