**Mathematical Model**

```
Indices
-------
I  – set of Baby products (rows of Salesdata.csv whose Product Name starts with “Baby”).

Parameters
----------
Revenue_i   – revenue per unit of product i  (column Revenue).
Demand_i    – demand for product i           (column Demand).
Initial_i   – initial inventory of product i (column Initial Inventory).

Decision Variables
------------------
x_i ∈ ℤ_+   – units of product i to sell.

Objective
---------
max  ∑_{i∈I} Revenue_i · x_i

Constraints
-----------
0 ≤ x_i ≤ Demand_i      ∀ i∈I
0 ≤ x_i ≤ Initial_i     ∀ i∈I
```

**Data Mapping**

| Parameter | Source Table | Table ID | Columns | Filter |
|-----------|--------------|----------|---------|--------|
| Revenue_i | Salesdata.csv | file_0_view_0 | Revenue | Product Name prefix “Baby” |
| Demand_i  | Salesdata.csv | file_0_view_0 | Demand  | Product Name prefix “Baby” |
| Initial_i | Salesdata.csv | file_0_view_0 | Initial Inventory | Product Name prefix “Baby” |

The index set `I` is the set of rows that satisfy the filter condition. Each parameter value is taken directly from the corresponding column of the filtered row. No additional numerical values are introduced beyond those supplied in the source data.