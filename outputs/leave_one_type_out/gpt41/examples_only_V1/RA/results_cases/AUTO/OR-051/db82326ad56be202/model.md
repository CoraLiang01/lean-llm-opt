Let x_{i,j} = number of units of coffee product j to be placed in cabinet i, where i indexes cabinets (CabinetID from capacity.csv) and j indexes products (row order from products.csv). All x_{i,j} are nonnegative integers.

Indices:
- i ∈ {1,2,3,4,5,6,7,8,9,10} (CabinetID from capacity.csv, in order)
- j ∈ {1,...,18} (products in order from products.csv)

Parameters:
- Capacity_i: capacity of cabinet i (from capacity.csv)
- Value_j: value of product j (from products.csv)
- Weight_j: weight of product j (from products.csv)

Data (in original order):

capacity.csv:
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

products.csv:
j | ProductName        | Value | Weight
1 | Espresso Beans     | 100   | 1.0
2 | Colombian Roast    | 150   | 1.5
3 | Arabica Blend      | 80    | 1.2
4 | French Roast       | 120   | 1.3
5 | Italian Roast      | 130   | 1.4
6 | House Blend        | 110   | 1.1
7 | Sumatra Coffee     | 160   | 1.8
8 | Mocha Java         | 90    | 1.2
9 | Hazelnut Flavor    | 95    | 1.0
10| Caramel Blend      | 105   | 1.3
11| Vanilla Flavor     | 85    | 1.2
12| Cappuccino Mix     | 140   | 1.5
13| Pumpkin Spice      | 75    | 1.1
14| Decaf Roast        | 60    | 1.0
15| Organic Roast      | 170   | 1.6
16| Cold Brew          | 115   | 1.4
17| Peruvian Blend     | 155   | 1.7
18| Kenyan AA          | 125   | 1.3

Decision variables:
x_{i,j} ∈ {0,1,2,...} for all i ∈ {1,...,10}, j ∈ {1,...,18}

Objective:
Maximize total value across all cabinets:
maximize
∑_{i=1}^{10} ∑_{j=1}^{18} Value_j * x_{i,j}
= ∑_{i=1}^{10} [100 x_{i,1} + 150 x_{i,2} + 80 x_{i,3} + 120 x_{i,4} + 130 x_{i,5} + 110 x_{i,6} + 160 x_{i,7} + 90 x_{i,8} + 95 x_{i,9} + 105 x_{i,10} + 85 x_{i,11} + 140 x_{i,12} + 75 x_{i,13} + 60 x_{i,14} + 170 x_{i,15} + 115 x_{i,16} + 155 x_{i,17} + 125 x_{i,18}]

Subject to (for each cabinet i, using its Capacity from capacity.csv):

For i = 1:
1.0 x_{1,1} + 1.5 x_{1,2} + 1.2 x_{1,3} + 1.3 x_{1,4} + 1.4 x_{1,5} + 1.1 x_{1,6} + 1.8 x_{1,7} + 1.2 x_{1,8} + 1.0 x_{1,9} + 1.3 x_{1,10} + 1.2 x_{1,11} + 1.5 x_{1,12} + 1.1 x_{1,13} + 1.0 x_{1,14} + 1.6 x_{1,15} + 1.4 x_{1,16} + 1.7 x_{1,17} + 1.3 x_{1,18} ≤ 400

For i = 2:
(same as above, with x_{2,j}, ≤ 600)

For i = 3:
(same as above, with x_{3,j}, ≤ 500)

For i = 4:
(same as above, with x_{4,j}, ≤ 700)

For i = 5:
(same as above, with x_{5,j}, ≤ 450)

For i = 6:
(same as above, with x_{6,j}, ≤ 650)

For i = 7:
(same as above, with x_{7,j}, ≤ 550)

For i = 8:
(same as above, with x_{8,j}, ≤ 750)

For i = 9:
(same as above, with x_{9,j}, ≤ 480)

For i = 10:
(same as above, with x_{10,j}, ≤ 520)

Variable domains:
x_{i,j} ∈ {0,1,2,...} (nonnegative integers), for all i ∈ {1,...,10}, j ∈ {1,...,18}

Summary:
Maximize total value of coffee products allocated to cabinets, subject to each cabinet's weight capacity, with integer numbers of units per product per cabinet. All data and indices are preserved in original order.