##### Model

**Index sets**  
\(W\) – set of warehouses (indexed by *Suppliers* in *supply_capacity.csv*)  
\(S\) – set of stores (indexed by *Customers* in *customer_demand.csv*)

**Parameters**  
\(D_s\) – daily demand at store \(s\) (from column *demand* in *customer_demand.csv*)  
\(C_w\) – supply capacity of warehouse \(w\) (from column *supply_capacity* in *supply_capacity.csv*)  
\(c_{w,s}\) – transportation cost per unit from warehouse \(w\) to store \(s\) (from column *CustomerX* in *transportation_costs.csv*, where *X* is the store identifier)

**Decision variables**  
\(x_{w,s}\) – quantity of produce shipped from warehouse \(w\) to store \(s\)  
\(x_{w,s} \ge 0\) for all \(w \in W,\; s \in S\)

**Objective function**  
\[
\min \;\sum_{w \in W}\;\sum_{s \in S}\; c_{w,s}\; x_{w,s}
\]

**Constraints**

1. **Demand satisfaction**  
\[
\sum_{w \in W} x_{w,s} \;\ge\; D_s, \qquad \forall s \in S
\]

2. **Warehouse capacity**  
\[
\sum_{s \in S} x_{w,s} \;\le\; C_w, \qquad \forall w \in W
\]

##### Data Mapping

| Parameter | Source Table | Exact Column Name |
|-----------|--------------|-------------------|
| \(D_s\)   | file_0_view_0 | demand |
| \(C_w\)   | file_1_view_0 | supply_capacity |
| \(c_{w,s}\) | file_2_view_0 | Customer1, Customer2, Customer3, Customer4, Customer5, Customer6 (columns corresponding to each store) |

*All tables are referenced by their exact file identifiers and column names as shown above.*