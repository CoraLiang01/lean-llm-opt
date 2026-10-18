Let:
- \( x_{ij} \): integer, number of units of product \( j \) (ProductName) placed in cabinet \( i \) (CabinetID), for all cabinets \( i \in \{1,2,3,4,5,6,7,8,9,10\} \) and all products \( j \) as listed below.

Parameters:
- Cabinets and their capacities (from capacity.csv, in file order):

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

- Products, their values and weights (from products.csv, in file order):

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

Model:

Variables:
- For each CabinetID \( i \in \{1,2,3,4,5,6,7,8,9,10\} \) and each ProductName \( j \) as listed above:
  - \( x_{ij} \in \mathbb{Z}_{\geq 0} \) (nonnegative integer)

Objective:
\[
\text{Maximize} \quad Z = \sum_{i=1}^{10} \sum_{j=1}^{18} \text{Value}_j \cdot x_{ij}
\]
where \(\text{Value}_j\) is the value of product \(j\) as listed above.

Subject to (for each cabinet \(i\)):
\[
\sum_{j=1}^{18} \text{Weight}_j \cdot x_{ij} \leq \text{Capacity}_i \qquad \forall i \in \{1,2,3,4,5,6,7,8,9,10\}
\]
where \(\text{Weight}_j\) is the weight of product \(j\) and \(\text{Capacity}_i\) is the capacity of cabinet \(i\) as listed above.

Variable domains:
\[
x_{ij} \in \{0,1,2,\ldots\} \qquad \forall i \in \{1,2,3,4,5,6,7,8,9,10\},\ \forall j \in \{\text{all 18 ProductNames above}\}
\]

This model uses all provided data, keeps the original file and row order, and aligns all indices and coefficients as required.