Let:
- i index cabinets, with CabinetID ∈ {1,2,3,4,5,6,7,8,9,10} (from capacity.csv, in order)
- j index products, with ProductName as below (from products.csv, in order)

Decision variables:
x_ij = number of units of product j placed in cabinet i, for all i and j
x_ij ∈ {0, 1, 2, ...} (nonnegative integers)

Parameters:
Cabinet capacities (from capacity.csv, in order):
CabinetID | Capacity
1         | 400
2         | 600
3         | 500
4         | 700
5         | 450
6         | 650
7         | 550
8         | 750
9         | 480
10        | 520

Product values and weights (from products.csv, in order):
j | ProductName         | Value | Weight
1 | Espresso Beans      | 100   | 1.0
2 | Colombian Roast     | 150   | 1.5
3 | Arabica Blend       | 80    | 1.2
4 | French Roast        | 120   | 1.3
5 | Italian Roast       | 130   | 1.4
6 | House Blend         | 110   | 1.1
7 | Sumatra Coffee      | 160   | 1.8
8 | Mocha Java          | 90    | 1.2
9 | Hazelnut Flavor     | 95    | 1.0
10| Caramel Blend       | 105   | 1.3
11| Vanilla Flavor      | 85    | 1.2
12| Cappuccino Mix      | 140   | 1.5
13| Pumpkin Spice       | 75    | 1.1
14| Decaf Roast         | 60    | 1.0
15| Organic Roast       | 170   | 1.6
16| Cold Brew           | 115   | 1.4
17| Peruvian Blend      | 155   | 1.7
18| Kenyan AA           | 125   | 1.3

Mathematical Model:

Variables:
For each cabinet i ∈ {1,...,10} and each product j ∈ {1,...,18}:
 x_ij ≥ 0 and integer

Objective:
Maximize total value across all cabinets:
 maximize ∑_{i=1}^{10} ∑_{j=1}^{18} Value_j * x_ij
  = ∑_{i=1}^{10} [100 x_{i,1} + 150 x_{i,2} + 80 x_{i,3} + 120 x_{i,4} + 130 x_{i,5} + 110 x_{i,6} + 160 x_{i,7} + 90 x_{i,8} + 95 x_{i,9} + 105 x_{i,10} + 85 x_{i,11} + 140 x_{i,12} + 75 x_{i,13} + 60 x_{i,14} + 170 x_{i,15} + 115 x_{i,16} + 155 x_{i,17} + 125 x_{i,18}]

Subject to (for each cabinet i):
 ∑_{j=1}^{18} Weight_j * x_ij ≤ Capacity_i

Explicitly, for each cabinet:

For CabinetID 1 (Capacity 400):
 1.0 x_{1,1} + 1.5 x_{1,2} + 1.2 x_{1,3} + 1.3 x_{1,4} + 1.4 x_{1,5} + 1.1 x_{1,6} + 1.8 x_{1,7} + 1.2 x_{1,8} + 1.0 x_{1,9} + 1.3 x_{1,10} + 1.2 x_{1,11} + 1.5 x_{1,12} + 1.1 x_{1,13} + 1.0 x_{1,14} + 1.6 x_{1,15} + 1.4 x_{1,16} + 1.7 x_{1,17} + 1.3 x_{1,18} ≤ 400

For CabinetID 2 (Capacity 600):
 1.0 x_{2,1} + 1.5 x_{2,2} + 1.2 x_{2,3} + 1.3 x_{2,4} + 1.4 x_{2,5} + 1.1 x_{2,6} + 1.8 x_{2,7} + 1.2 x_{2,8} + 1.0 x_{2,9} + 1.3 x_{2,10} + 1.2 x_{2,11} + 1.5 x_{2,12} + 1.1 x_{2,13} + 1.0 x_{2,14} + 1.6 x_{2,15} + 1.4 x_{2,16} + 1.7 x_{2,17} + 1.3 x_{2,18} ≤ 600

For CabinetID 3 (Capacity 500):
 1.0 x_{3,1} + 1.5 x_{3,2} + 1.2 x_{3,3} + 1.3 x_{3,4} + 1.4 x_{3,5} + 1.1 x_{3,6} + 1.8 x_{3,7} + 1.2 x_{3,8} + 1.0 x_{3,9} + 1.3 x_{3,10} + 1.2 x_{3,11} + 1.5 x_{3,12} + 1.1 x_{3,13} + 1.0 x_{3,14} + 1.6 x_{3,15} + 1.4 x_{3,16} + 1.7 x_{3,17} + 1.3 x_{3,18} ≤ 500

For CabinetID 4 (Capacity 700):
 1.0 x_{4,1} + 1.5 x_{4,2} + 1.2 x_{4,3} + 1.3 x_{4,4} + 1.4 x_{4,5} + 1.1 x_{4,6} + 1.8 x_{4,7} + 1.2 x_{4,8} + 1.0 x_{4,9} + 1.3 x_{4,10} + 1.2 x_{4,11} + 1.5 x_{4,12} + 1.1 x_{4,13} + 1.0 x_{4,14} + 1.6 x_{4,15} + 1.4 x_{4,16} + 1.7 x_{4,17} + 1.3 x_{4,18} ≤ 700

For CabinetID 5 (Capacity 450):
 1.0 x_{5,1} + 1.5 x_{5,2} + 1.2 x_{5,3} + 1.3 x_{5,4} + 1.4 x_{5,5} + 1.1 x_{5,6} + 1.8 x_{5,7} + 1.2 x_{5,8} + 1.0 x_{5,9} + 1.3 x_{5,10} + 1.2 x_{5,11} + 1.5 x_{5,12} + 1.1 x_{5,13} + 1.0 x_{5,14} + 1.6 x_{5,15} + 1.4 x_{5,16} + 1.7 x_{5,17} + 1.3 x_{5,18} ≤ 450

For CabinetID 6 (Capacity 650):
 1.0 x_{6,1} + 1.5 x_{6,2} + 1.2 x_{6,3} + 1.3 x_{6,4} + 1.4 x_{6,5} + 1.1 x_{6,6} + 1.8 x_{6,7} + 1.2 x_{6,8} + 1.0 x_{6,9} + 1.3 x_{6,10} + 1.2 x_{6,11} + 1.5 x_{6,12} + 1.1 x_{6,13} + 1.0 x_{6,14} + 1.6 x_{6,15} + 1.4 x_{6,16} + 1.7 x_{6,17} + 1.3 x_{6,18} ≤ 650

For CabinetID 7 (Capacity 550):
 1.0 x_{7,1} + 1.5 x_{7,2} + 1.2 x_{7,3} + 1.3 x_{7,4} + 1.4 x_{7,5} + 1.1 x_{7,6} + 1.8 x_{7,7} + 1.2 x_{7,8} + 1.0 x_{7,9} + 1.3 x_{7,10} + 1.2 x_{7,11} + 1.5 x_{7,12} + 1.1 x_{7,13} + 1.0 x_{7,14} + 1.6 x_{7,15} + 1.4 x_{7,16} + 1.7 x_{7,17} + 1.3 x_{7,18} ≤ 550

For CabinetID 8 (Capacity 750):
 1.0 x_{8,1} + 1.5 x_{8,2} + 1.2 x_{8,3} + 1.3 x_{8,4} + 1.4 x_{8,5} + 1.1 x_{8,6} + 1.8 x_{8,7} + 1.2 x_{8,8} + 1.0 x_{8,9} + 1.3 x_{8,10} + 1.2 x_{8,11} + 1.5 x_{8,12} + 1.1 x_{8,13} + 1.0 x_{8,14} + 1.6 x_{8,15} + 1.4 x_{8,16} + 1.7 x_{8,17} + 1.3 x_{8,18} ≤ 750

For CabinetID 9 (Capacity 480):
 1.0 x_{9,1} + 1.5 x_{9,2} + 1.2 x_{9,3} + 1.3 x_{9,4} + 1.4 x_{9,5} + 1.1 x_{9,6} + 1.8 x_{9,7} + 1.2 x_{9,8} + 1.0 x_{9,9} + 1.3 x_{9,10} + 1.2 x_{9,11} + 1.5 x_{9,12} + 1.1 x_{9,13} + 1.0 x_{9,14} + 1.6 x_{9,15} + 1.4 x_{9,16} + 1.7 x_{9,17} + 1.3 x_{9,18} ≤ 480

For CabinetID 10 (Capacity 520):
 1.0 x_{10,1} + 1.5 x_{10,2} + 1.2 x_{10,3} + 1.3 x_{10,4} + 1.4 x_{10,5} + 1.1 x_{10,6} + 1.8 x_{10,7} + 1.2 x_{10,8} + 1.0 x_{10,9} + 1.3 x_{10,10} + 1.2 x_{10,11} + 1.5 x_{10,12} + 1.1 x_{10,13} + 1.0 x_{10,14} + 1.6 x_{10,15} + 1.4 x_{10,16} + 1.7 x_{10,17} + 1.3 x_{10,18} ≤ 520

Variable domains:
x_ij ∈ {0, 1, 2, ...} for all i ∈ {1,...,10}, j ∈ {1,...,18}

This is a complete integer programming formulation for the described allocation problem, using all provided data and preserving file and row order.