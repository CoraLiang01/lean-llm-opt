#### Abstract Mathematical Model

Let:
- $\mathcal{I}$: index set of vehicle types (from products.csv and capacity.csv)
- $b_i$: benefit coefficient for vehicle type $i \in \mathcal{I}$ (from products.csv, column "Value")
- $c_i$: daily inventory limit for vehicle type $i \in \mathcal{I}$ (from capacity.csv, column "Capacity")
- $C^{\text{tot}}$: total inventory capacity per day (from capacity.csv, sum of "Capacity" column or as specified)
- $x_i$: integer decision variable, number of vehicles of type $i$ to order per day

Objective:
$$
\max \sum_{i \in \mathcal{I}} b_i x_i
$$

Subject to:
1. Vehicle-type daily inventory limits:
$$
x_i \leq c_i \quad \forall i \in \mathcal{I}
$$

2. Total inventory capacity:
$$
\sum_{i \in \mathcal{I}} x_i \leq C^{\text{tot}}
$$

3. Integer and non-negativity constraints:
$$
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in \mathcal{I}
$$

#### Data Mapping

- Table: capacity.csv (table_id: file_0_view_0)
  - VehicleType $\rightarrow$ index set $\mathcal{I}$
  - Capacity $\rightarrow$ parameter $c_i$ for $i \in \mathcal{I}$
  - (Sum of Capacity column or explicit field) $\rightarrow$ $C^{\text{tot}}$

- Table: products.csv (table_id: file_1_view_0)
  - ProductName $\rightarrow$ index set $\mathcal{I}$
  - Value $\rightarrow$ parameter $b_i$ for $i \in \mathcal{I}$

No literal data values or record counts are included; all sets and parameters are defined symbolically and mapped to their source columns.