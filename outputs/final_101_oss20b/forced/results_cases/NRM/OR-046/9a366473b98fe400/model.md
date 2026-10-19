##### Index Sets  
\(I\) – set of all products.

##### Parameters  
\(p_i\) – profit per unit of product \(i\) (from *products.csv*, column *Value*).  
\(C\) – total stock capacity (from *capacity.csv*, column *Capacity*).

##### Decision Variables  
\(x_i\) – number of units of product \(i\) to order each day.  
\(x_i \in \mathbb{Z}_{\ge 0}\).

##### Objective Function  
$$
\max \sum_{i \in I} p_i \, x_i
$$

##### Constraints  

###### 1. Capacity Constraint  
$$
\sum_{i \in I} x_i \;\le\; C
$$

##### Data Mapping  

- **products.csv** – table_id: `file_1_view_0`; columns: `ProductName`, `Value`.  
- **capacity.csv** – table_id: `file_0_view_0`; column: `Capacity`.