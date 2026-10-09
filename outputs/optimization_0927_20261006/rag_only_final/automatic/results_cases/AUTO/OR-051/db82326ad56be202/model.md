Let:
- i index cabinets, with CabinetID ∈ {1,2,3,4,5,6,7,8,9,10}
- j index products, with ProductName as below

Parameters:
- Capacity_i: capacity of cabinet i (from capacity.csv)
- Value_j: value per unit of product j (from products.csv)
- Weight_j: weight per unit of product j (from products.csv)

Decision variables:
- x_ij: number of units of product j to place in cabinet i (integer, x_ij ≥ 0)

Data:
Cabinets (from capacity.csv, in file order):
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

Products (from products.csv, in file order):
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

Mathematical Model:

Variables:
For each cabinet i ∈ {1,...,10} and each product j ∈ {1,...,18} (in the order above),
 x_ij ∈ {0,1,2,...}

Objective:
Maximize total value across all cabinets:
\[
\text{Maximize} \quad Z = \sum_{i=1}^{10} \sum_{j=1}^{18} \text{Value}_j \cdot x_{ij}
\]
where Value_j is as listed above for each product.

Subject to:

For each cabinet i (CabinetID as above):
\[
\sum_{j=1}^{18} \text{Weight}_j \cdot x_{ij} \leq \text{Capacity}_i
\]
where Weight_j and Capacity_i are as listed above.

Variable domains:
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in \{1,\ldots,10\},\ j \in \{1,\ldots,18\}
\]

All data and indices are as provided in the original files and order. No additional constraints or data are assumed.