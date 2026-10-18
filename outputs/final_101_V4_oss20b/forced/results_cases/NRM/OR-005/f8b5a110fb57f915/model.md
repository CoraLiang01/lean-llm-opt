##### Index Sets  
- \(D\): set of distribution centers.  
- \(C\): set of customer groups.  

##### Parameters  
- \(demand_c\): daily demand of customer group \(c \in C\).  
- \(capacity_d\): daily supply capacity of distribution center \(d \in D\).  
- \(cost_{dc}\): transportation cost per unit from distribution center \(d\) to customer group \(c\).  

##### Variables  
- \(x_{dc}\): quantity of goods shipped from distribution center \(d\) to customer group \(c\), \(x_{dc} \ge 0\).  

##### Objective Function  
\[
\min \sum_{d \in D} \sum_{c \in C} cost_{dc} \cdot x_{dc}
\]

##### Constraints  

###### 1. Supply Capacity Constraints  
\[
\sum_{c \in C} x_{dc} \le capacity_d, \quad \forall d \in D
\]

###### 2. Demand Fulfillment Constraints  
\[
\sum_{d \in D} x_{dc} \ge demand_c, \quad \forall c \in C
\]

###### 3. Variable Non‑negativity  
\[
x_{dc} \ge 0, \quad \forall d \in D, \forall c \in C
\]

##### Data Mapping  

- **customer_demand.csv**  
  - Column `Customers` → index set \(C\)  
  - Column `demand` → parameter \(demand_c\)

- **supply_capacity.csv**  
  - Column `Supplier` → index set \(D\)  
  - Column `supply_capacity` → parameter \(capacity_d\)

- **transportation_costs.csv**  
  - Column `Unnamed: 0` → index set \(D\)  
  - Columns `demand1` … `demand8` → customer groups in \(C\)  
  - Each column `demandk` → parameter \(cost_{dc}\) for the corresponding customer group \(c\) and distribution center \(d\)