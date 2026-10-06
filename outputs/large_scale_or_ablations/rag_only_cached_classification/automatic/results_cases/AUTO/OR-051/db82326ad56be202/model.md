Let:
- i index the cabinets, with CabinetID from capacity.csv: i ∈ {1,2,3,4,5,6,7,8,9,10}
- j index the coffee products, with ProductName from products.csv: j ∈ {Espresso Beans, Colombian Roast, Arabica Blend, French Roast, Italian Roast, House Blend, Sumatra Coffee, Mocha Java, Hazelnut Flavor, Caramel Blend, Vanilla Flavor, Cappuccino Mix, Pumpkin Spice, Decaf Roast, Organic Roast, Cold Brew, Peruvian Blend, Kenyan AA}
- x_{i,j} = number of units of product j placed in cabinet i (integer, x_{i,j} ≥ 0)

Parameters:
- Capacity_i: capacity of cabinet i (from capacity.csv)
- Value_j: value per unit of product j (from products.csv)
- Weight_j: weight per unit of product j (from products.csv)

Data:
capacity.csv

| CabinetID | Capacity |
|-----------|----------|
| 1         | 400      |
| 2         | 600      |
| 3         | 500      |
| 4         | 700      |
| 5         | 450      |
| 6         | 650      |
| 7         | 550      |
| 8         | 750      |
| 9         | 480      |
| 10        | 520      |

products.csv

| ProductName        | Value | Weight |
|--------------------|-------|--------|
| Espresso Beans     | 100   | 1.0    |
| Colombian Roast    | 150   | 1.5    |
| Arabica Blend      | 80    | 1.2    |
| French Roast       | 120   | 1.3    |
| Italian Roast      | 130   | 1.4    |
| House Blend        | 110   | 1.1    |
| Sumatra Coffee     | 160   | 1.8    |
| Mocha Java         | 90    | 1.2    |
| Hazelnut Flavor    | 95    | 1.0    |
| Caramel Blend      | 105   | 1.3    |
| Vanilla Flavor     | 85    | 1.2    |
| Cappuccino Mix     | 140   | 1.5    |
| Pumpkin Spice      | 75    | 1.1    |
| Decaf Roast        | 60    | 1.0    |
| Organic Roast      | 170   | 1.6    |
| Cold Brew          | 115   | 1.4    |
| Peruvian Blend     | 155   | 1.7    |
| Kenyan AA          | 125   | 1.3    |

Mathematical Optimization Model:

Decision Variables:
x_{i,j} ∈ {0, 1, 2, ...} for all i ∈ {1,...,10}, j ∈ {1,...,18}

Objective:
Maximize total value of products placed in all cabinets:
\[
\text{Maximize} \quad Z = \sum_{i=1}^{10} \sum_{j=1}^{18} \text{Value}_j \cdot x_{i,j}
\]
where Value_j is as given in products.csv, in the original row order.

Subject to:

For each cabinet i (CabinetID as in capacity.csv, in original order):
\[
\sum_{j=1}^{18} \text{Weight}_j \cdot x_{i,j} \leq \text{Capacity}_i
\]
where Weight_j is as given in products.csv, in the original row order, and Capacity_i is as given in capacity.csv.

Variable domains:
\[
x_{i,j} \in \mathbb{Z}_{\geq 0} \quad \forall i \in \{1,\ldots,10\},\ j \in \{1,\ldots,18\}
\]

Explicitly, for each CabinetID i ∈ {1,2,...,10} and each ProductName j in the order above, the model is:

Maximize:
\[
Z = \sum_{i=1}^{10} \Big(
100\,x_{i,1} + 150\,x_{i,2} + 80\,x_{i,3} + 120\,x_{i,4} + 130\,x_{i,5} + 110\,x_{i,6} + 160\,x_{i,7} + 90\,x_{i,8} + 95\,x_{i,9} + 105\,x_{i,10} + 85\,x_{i,11} + 140\,x_{i,12} + 75\,x_{i,13} + 60\,x_{i,14} + 170\,x_{i,15} + 115\,x_{i,16} + 155\,x_{i,17} + 125\,x_{i,18}
\Big)
\]

Subject to, for each i = 1,...,10 (CabinetID as in capacity.csv):

\[
1.0\,x_{i,1} + 1.5\,x_{i,2} + 1.2\,x_{i,3} + 1.3\,x_{i,4} + 1.4\,x_{i,5} + 1.1\,x_{i,6} + 1.8\,x_{i,7} + 1.2\,x_{i,8} + 1.0\,x_{i,9} + 1.3\,x_{i,10} + 1.2\,x_{i,11} + 1.5\,x_{i,12} + 1.1\,x_{i,13} + 1.0\,x_{i,14} + 1.6\,x_{i,15} + 1.4\,x_{i,16} + 1.7\,x_{i,17} + 1.3\,x_{i,18} \leq \text{Capacity}_i
\]

where Capacity_i is the value from capacity.csv for CabinetID i.

All x_{i,j} are integer and nonnegative.

This model uses all provided data, preserves row order, and aligns all indices and coefficients as required.