Let:
- i index the storage areas, with StorageID i ∈ {1, 2, ..., 15} as given in capacity.csv.
- j index the air conditioner types, with ProductName j ∈ {"Window Unit", "Portable Unit", "Split System", "Ductless System", "Central AC", "Hybrid AC", "Geothermal AC", "Smart AC", "Evaporative Cooler", "Package Unit"} as given in products.csv.
- x_{i,j} = number of units of air conditioner type j to be placed in storage area i (integer, x_{i,j} ≥ 0).

Parameters:
From capacity.csv:
- Capacity_i = capacity of storage area i.

From products.csv:
- Value_j = value of air conditioner type j.
- Weight_j = size (weight) of air conditioner type j.

Model:

Decision variables:
x_{i,j} ∈ {0, 1, 2, ...} for all i ∈ {1,...,15}, j ∈ {1,...,10}

Objective:
Maximize total value of air conditioners allocated:
\[
\text{Maximize} \quad Z = \sum_{i=1}^{15} \sum_{j=1}^{10} \text{Value}_j \cdot x_{i,j}
\]
where Value_j is as follows (from products.csv, in order):
1. Window Unit: 4811
2. Portable Unit: 1130
3. Split System: 1611
4. Ductless System: 3368
5. Central AC: 2135
6. Hybrid AC: 1046
7. Geothermal AC: 4030
8. Smart AC: 3761
9. Evaporative Cooler: 3523
10. Package Unit: 1701

Subject to:

For each storage area i (StorageID from capacity.csv):

\[
\sum_{j=1}^{10} \text{Weight}_j \cdot x_{i,j} \leq \text{Capacity}_i
\]
where Weight_j is as follows (from products.csv, in order):
1. Window Unit: 114
2. Portable Unit: 200
3. Split System: 106
4. Ductless System: 256
5. Central AC: 268
6. Hybrid AC: 185
7. Geothermal AC: 299
8. Smart AC: 131
9. Evaporative Cooler: 139
10. Package Unit: 105

And Capacity_i is as follows (from capacity.csv, in order):
1. StorageID 1: 1083
2. StorageID 2: 1840
3. StorageID 3: 770
4. StorageID 4: 1299
5. StorageID 5: 1259
6. StorageID 6: 543
7. StorageID 7: 1831
8. StorageID 8: 855
9. StorageID 9: 619
10. StorageID 10: 637
11. StorageID 11: 935
12. StorageID 12: 626
13. StorageID 13: 1457
14. StorageID 14: 1198
15. StorageID 15: 837

So, for each i ∈ {1,...,15}:
\[
114\,x_{i,1} + 200\,x_{i,2} + 106\,x_{i,3} + 256\,x_{i,4} + 268\,x_{i,5} + 185\,x_{i,6} + 299\,x_{i,7} + 131\,x_{i,8} + 139\,x_{i,9} + 105\,x_{i,10} \leq \text{Capacity}_i
\]

Variable domains:
\[
x_{i,j} \in \mathbb{Z}_{\geq 0} \quad \forall i \in \{1,...,15\},\ j \in \{1,...,10\}
\]

Summary:
- Decision variables: x_{i,j} = number of units of air conditioner type j in storage area i (integer, ≥0)
- Objective: Maximize total value allocated
- Constraints: For each storage area, total size of allocated units ≤ its capacity
- All coefficients and indices are as given in the supplied CSVs.