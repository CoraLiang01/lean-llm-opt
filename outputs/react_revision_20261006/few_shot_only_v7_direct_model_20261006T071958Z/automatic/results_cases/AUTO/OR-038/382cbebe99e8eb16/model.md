ABSTRACT MATHEMATICAL MODEL

Sets:
- $I$: set of vehicle types, indexed by $i$ (from all ProductName in file_1_view_0)

Parameters:
- $b_i$: benefit coefficient of vehicle type $i$ (from Value in file_1_view_0, mapped by ProductName)
- $c_i$: daily inventory capacity for vehicle type $i$ (from Capacity in file_0_view_0, mapped by VehicleType)
- $w_i$: weight per unit of vehicle type $i$ (from Weight in file_1_view_0, mapped by ProductName)

Decision Variables:
- $x_i \in \mathbb{Z}_{\geq 0}$: number of vehicles of type $i$ to order per day

Objective:
\[
\max \sum_{i \in I} b_i x_i
\]

Constraints:
\[
x_i \leq c_i \qquad \forall i \in I
\]
\[
x_i \in \mathbb{Z}_{\geq 0} \qquad \forall i \in I
\]

Data Mapping:

- $I$: All ProductName in file_1_view_0
- $b_i$: file_1_view_0, column Value, keyed by ProductName
- $c_i$: file_0_view_0, column Capacity, keyed by VehicleType (matched to ProductName)
- $w_i$: file_1_view_0, column Weight, keyed by ProductName
- $x_i$: integer variable for each $i \in I$