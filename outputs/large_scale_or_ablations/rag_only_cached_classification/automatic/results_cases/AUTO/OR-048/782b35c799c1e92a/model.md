Let:
- I = {1, 2, ..., 15} be the set of storage areas, indexed by i (from StorageID in capacity.csv)
- J = {1, 2, ..., 10} be the set of air conditioner types, indexed by j (corresponding to the row order in products.csv)

Let:
- Capacity_i = capacity of storage area i (from capacity.csv)
- Value_j = value of air conditioner type j (from products.csv)
- Weight_j = weight (size) of air conditioner type j (from products.csv)
- x_ij = integer number of units of air conditioner type j placed in storage area i, for all i in I, j in J

Variables:
x_ij ≥ 0 and integer, ∀ i ∈ I, j ∈ J

Data (in original order):

capacity.csv:
| StorageID (i) | Capacity_i |
|---------------|------------|
| 1             | 1083       |
| 2             | 1840       |
| 3             | 770        |
| 4             | 1299       |
| 5             | 1259       |
| 6             | 543        |
| 7             | 1831       |
| 8             | 855        |
| 9             | 619        |
| 10            | 637        |
| 11            | 935        |
| 12            | 626        |
| 13            | 1457       |
| 14            | 1198       |
| 15            | 837        |

products.csv (j = 1 to 10 in this order):
| j | ProductName         | Value_j | Weight_j |
|---|---------------------|---------|----------|
| 1 | Window Unit         | 4811    | 114      |
| 2 | Portable Unit       | 1130    | 200      |
| 3 | Split System        | 1611    | 106      |
| 4 | Ductless System     | 3368    | 256      |
| 5 | Central AC          | 2135    | 268      |
| 6 | Hybrid AC           | 1046    | 185      |
| 7 | Geothermal AC       | 4030    | 299      |
| 8 | Smart AC            | 3761    | 131      |
| 9 | Evaporative Cooler  | 3523    | 139      |
|10 | Package Unit        | 1701    | 105      |

Mathematical Optimization Model:

Variables:
x_ij ∈ {0, 1, 2, ...} for i = 1,...,15; j = 1,...,10

Objective:
Maximize Z = ∑_{i=1}^{15} ∑_{j=1}^{10} Value_j * x_ij
      = ∑_{i=1}^{15} [4811 x_{i,1} + 1130 x_{i,2} + 1611 x_{i,3} + 3368 x_{i,4} + 2135 x_{i,5} + 1046 x_{i,6} + 4030 x_{i,7} + 3761 x_{i,8} + 3523 x_{i,9} + 1701 x_{i,10}]

Subject to (for each storage area i, using its StorageID and Capacity_i):

For i = 1:
 114 x_{1,1} + 200 x_{1,2} + 106 x_{1,3} + 256 x_{1,4} + 268 x_{1,5} + 185 x_{1,6} + 299 x_{1,7} + 131 x_{1,8} + 139 x_{1,9} + 105 x_{1,10} ≤ 1083

For i = 2:
 114 x_{2,1} + 200 x_{2,2} + 106 x_{2,3} + 256 x_{2,4} + 268 x_{2,5} + 185 x_{2,6} + 299 x_{2,7} + 131 x_{2,8} + 139 x_{2,9} + 105 x_{2,10} ≤ 1840

For i = 3:
 114 x_{3,1} + 200 x_{3,2} + 106 x_{3,3} + 256 x_{3,4} + 268 x_{3,5} + 185 x_{3,6} + 299 x_{3,7} + 131 x_{3,8} + 139 x_{3,9} + 105 x_{3,10} ≤ 770

For i = 4:
 114 x_{4,1} + 200 x_{4,2} + 106 x_{4,3} + 256 x_{4,4} + 268 x_{4,5} + 185 x_{4,6} + 299 x_{4,7} + 131 x_{4,8} + 139 x_{4,9} + 105 x_{4,10} ≤ 1299

For i = 5:
 114 x_{5,1} + 200 x_{5,2} + 106 x_{5,3} + 256 x_{5,4} + 268 x_{5,5} + 185 x_{5,6} + 299 x_{5,7} + 131 x_{5,8} + 139 x_{5,9} + 105 x_{5,10} ≤ 1259

For i = 6:
 114 x_{6,1} + 200 x_{6,2} + 106 x_{6,3} + 256 x_{6,4} + 268 x_{6,5} + 185 x_{6,6} + 299 x_{6,7} + 131 x_{6,8} + 139 x_{6,9} + 105 x_{6,10} ≤ 543

For i = 7:
 114 x_{7,1} + 200 x_{7,2} + 106 x_{7,3} + 256 x_{7,4} + 268 x_{7,5} + 185 x_{7,6} + 299 x_{7,7} + 131 x_{7,8} + 139 x_{7,9} + 105 x_{7,10} ≤ 1831

For i = 8:
 114 x_{8,1} + 200 x_{8,2} + 106 x_{8,3} + 256 x_{8,4} + 268 x_{8,5} + 185 x_{8,6} + 299 x_{8,7} + 131 x_{8,8} + 139 x_{8,9} + 105 x_{8,10} ≤ 855

For i = 9:
 114 x_{9,1} + 200 x_{9,2} + 106 x_{9,3} + 256 x_{9,4} + 268 x_{9,5} + 185 x_{9,6} + 299 x_{9,7} + 131 x_{9,8} + 139 x_{9,9} + 105 x_{9,10} ≤ 619

For i = 10:
 114 x_{10,1} + 200 x_{10,2} + 106 x_{10,3} + 256 x_{10,4} + 268 x_{10,5} + 185 x_{10,6} + 299 x_{10,7} + 131 x_{10,8} + 139 x_{10,9} + 105 x_{10,10} ≤ 637

For i = 11:
 114 x_{11,1} + 200 x_{11,2} + 106 x_{11,3} + 256 x_{11,4} + 268 x_{11,5} + 185 x_{11,6} + 299 x_{11,7} + 131 x_{11,8} + 139 x_{11,9} + 105 x_{11,10} ≤ 935

For i = 12:
 114 x_{12,1} + 200 x_{12,2} + 106 x_{12,3} + 256 x_{12,4} + 268 x_{12,5} + 185 x_{12,6} + 299 x_{12,7} + 131 x_{12,8} + 139 x_{12,9} + 105 x_{12,10} ≤ 626

For i = 13:
 114 x_{13,1} + 200 x_{13,2} + 106 x_{13,3} + 256 x_{13,4} + 268 x_{13,5} + 185 x_{13,6} + 299 x_{13,7} + 131 x_{13,8} + 139 x_{13,9} + 105 x_{13,10} ≤ 1457

For i = 14:
 114 x_{14,1} + 200 x_{14,2} + 106 x_{14,3} + 256 x_{14,4} + 268 x_{14,5} + 185 x_{14,6} + 299 x_{14,7} + 131 x_{14,8} + 139 x_{14,9} + 105 x_{14,10} ≤ 1198

For i = 15:
 114 x_{15,1} + 200 x_{15,2} + 106 x_{15,3} + 256 x_{15,4} + 268 x_{15,5} + 185 x_{15,6} + 299 x_{15,7} + 131 x_{15,8} + 139 x_{15,9} + 105 x_{15,10} ≤ 837

And for all i, j:
 x_{i,j} ∈ {0, 1, 2, ...}

This is a complete integer programming formulation for the described allocation problem, using all provided data and preserving all original identifiers and row order.