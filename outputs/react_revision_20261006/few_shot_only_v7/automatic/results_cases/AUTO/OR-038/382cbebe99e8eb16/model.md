##### Mathematical Model

Let $I$ be the set of vehicle types, indexed by $i$.

**Parameters:**
- $b_i$: Benefit coefficient for vehicle type $i$ (from products.csv, column Value, keyed by ProductName)
- $c_i$: Inventory capacity for vehicle type $i$ (from capacity.csv, column Capacity, keyed by VehicleType)

**Decision Variables:**
- $x_i$: Number of vehicles of type $i$ to order per day; $x_i \in \mathbb{Z}_{\geq 0}$

**Objective:**
\[
\max \sum_{i \in I} b_i x_i
\]

**Constraints:**
\[
x_i \leq c_i \quad \forall i \in I
\]
\[
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
\]

##### Data Mapping

- $I$: Set of vehicle types, from /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA4/NewCarSalesInNorway1/capacity.csv, column VehicleType, and /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA4/NewCarSalesInNorway1/products.csv, column ProductName
- $b_i$: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA4/NewCarSalesInNorway1/products.csv, column Value, keyed by ProductName
- $c_i$: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA4/NewCarSalesInNorway1/capacity.csv, column Capacity, keyed by VehicleType
- $x_i$: Number of vehicles of type $i$ to order per day (decision variable)