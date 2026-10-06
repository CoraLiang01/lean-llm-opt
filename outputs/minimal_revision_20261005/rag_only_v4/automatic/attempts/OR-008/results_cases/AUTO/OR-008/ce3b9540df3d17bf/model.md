Sets:
- Let \( S \) be the set of suppliers (warehouses):  
  \( S = \{\text{Supplier1}, \text{Supplier2}, \text{Supplier3}, \text{Supplier4}, \text{Supplier5}\} \)  
  (from supply_capacity.csv, file_1_view_0, column "Suppliers")
- Let \( C \) be the set of customers (stores):  
  \( C = \{\text{Customer1}, \text{Customer2}, \text{Customer3}, \text{Customer4}, \text{Customer5}, \text{Customer6}\} \)  
  (from customer_demand.csv, file_0_view_0, column "Customers")

Parameters:
- \( d_j \): Demand of customer \( j \in C \)  
  (from file_0_view_0, column "demand", indexed by "Customers")
- \( u_i \): Supply capacity of supplier \( i \in S \)  
  (from file_1_view_0, column "supply_capacity", indexed by "Suppliers")
- \( c_{ij} \): Transportation cost per unit from supplier \( i \) to customer \( j \)  
  (from file_2_view_0, row "Unnamed: 0" = supplier, column = customer)

Decision Variables:
- \( x_{ij} \geq 0 \): Amount shipped from supplier \( i \) to customer \( j \), for all \( i \in S, j \in C \)

Model:

\[
\begin{align*}
\min_{x_{ij} \geq 0} \quad & \sum_{i \in S} \sum_{j \in C} c_{ij} x_{ij} \\
\text{s.t.} \quad
& \sum_{i \in S} x_{ij} = d_j \quad && \forall j \in C \\
& \sum_{j \in C} x_{ij} \leq u_i \quad && \forall i \in S \\
\end{align*}
\]

Data Mapping:

- \( S \): All "Suppliers" in file_1_view_0
- \( C \): All "Customers" in file_0_view_0
- \( d_j \): file_0_view_0, column "demand", indexed by "Customers"
- \( u_i \): file_1_view_0, column "supply_capacity", indexed by "Suppliers"
- \( c_{ij} \): file_2_view_0, row "Unnamed: 0" = supplier, column = customer

Variable domains:
- \( x_{ij} \geq 0 \), continuous, for all \( i \in S, j \in C \)

Objective:
- Minimize total transportation cost

Constraints:
- Each customer’s demand is exactly met
- No supplier exceeds its capacity

All indices, parameters, and coefficients are bound directly to the supplied data as described above.