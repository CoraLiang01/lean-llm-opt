Let:
- \( I = \{1,2,3,4,5,6,7,8,9,10\} \) be the set of cabinets, indexed by CabinetID as given in capacity.csv.
- \( J = \{\text{Espresso Beans}, \text{Colombian Roast}, \text{Arabica Blend}, \text{French Roast}, \text{Italian Roast}, \text{House Blend}, \text{Sumatra Coffee}, \text{Mocha Java}, \text{Hazelnut Flavor}, \text{Caramel Blend}, \text{Vanilla Flavor}, \text{Cappuccino Mix}, \text{Pumpkin Spice}, \text{Decaf Roast}, \text{Organic Roast}, \text{Cold Brew}, \text{Peruvian Blend}, \text{Kenyan AA}\} \) be the set of coffee products, indexed by ProductName as given in products.csv.

Parameters:
- For each cabinet \( i \in I \), let \( C_i \) be its capacity:
  - \( C_1 = 400 \)
  - \( C_2 = 600 \)
  - \( C_3 = 500 \)
  - \( C_4 = 700 \)
  - \( C_5 = 450 \)
  - \( C_6 = 650 \)
  - \( C_7 = 550 \)
  - \( C_8 = 750 \)
  - \( C_9 = 480 \)
  - \( C_{10} = 520 \)
- For each product \( j \in J \), let \( v_j \) be its value and \( w_j \) its weight per unit:

| ProductName         | \( v_j \) (Value) | \( w_j \) (Weight) |
|---------------------|-------------------|--------------------|
| Espresso Beans      | 100               | 1.0                |
| Colombian Roast     | 150               | 1.5                |
| Arabica Blend       | 80                | 1.2                |
| French Roast        | 120               | 1.3                |
| Italian Roast       | 130               | 1.4                |
| House Blend         | 110               | 1.1                |
| Sumatra Coffee      | 160               | 1.8                |
| Mocha Java          | 90                | 1.2                |
| Hazelnut Flavor     | 95                | 1.0                |
| Caramel Blend       | 105               | 1.3                |
| Vanilla Flavor      | 85                | 1.2                |
| Cappuccino Mix      | 140               | 1.5                |
| Pumpkin Spice       | 75                | 1.1                |
| Decaf Roast         | 60                | 1.0                |
| Organic Roast       | 170               | 1.6                |
| Cold Brew           | 115               | 1.4                |
| Peruvian Blend      | 155               | 1.7                |
| Kenyan AA           | 125               | 1.3                |

Decision variables:
- For each cabinet \( i \in I \) and product \( j \in J \), let \( x_{ij} \) be the number of units of product \( j \) to place in cabinet \( i \).
- \( x_{ij} \) are nonnegative integers: \( x_{ij} \in \mathbb{Z}_+, \forall i \in I, j \in J \).

Objective:
\[
\max \sum_{i \in I} \sum_{j \in J} v_j x_{ij}
\]
That is, maximize the total value of all products placed in all cabinets.

Constraints:
For each cabinet \( i \in I \):
\[
\sum_{j \in J} w_j x_{ij} \leq C_i
\]
That is, the total weight of products in cabinet \( i \) does not exceed its capacity.

Variable domains:
\[
x_{ij} \in \mathbb{Z}_+, \quad \forall i \in I, j \in J
\]

Full numerical model:

Let \( x_{ij} \) be the integer number of units of product \( j \) in cabinet \( i \).

\[
\begin{align*}
\max \quad & \sum_{i=1}^{10} \Big[ 
100\,x_{i,\text{Espresso Beans}} + 
150\,x_{i,\text{Colombian Roast}} + 
80\,x_{i,\text{Arabica Blend}} + 
120\,x_{i,\text{French Roast}} + 
130\,x_{i,\text{Italian Roast}} + 
110\,x_{i,\text{House Blend}} + 
160\,x_{i,\text{Sumatra Coffee}} + \\
& \qquad 90\,x_{i,\text{Mocha Java}} + 
95\,x_{i,\text{Hazelnut Flavor}} + 
105\,x_{i,\text{Caramel Blend}} + 
85\,x_{i,\text{Vanilla Flavor}} + 
140\,x_{i,\text{Cappuccino Mix}} + 
75\,x_{i,\text{Pumpkin Spice}} + \\
& \qquad 60\,x_{i,\text{Decaf Roast}} + 
170\,x_{i,\text{Organic Roast}} + 
115\,x_{i,\text{Cold Brew}} + 
155\,x_{i,\text{Peruvian Blend}} + 
125\,x_{i,\text{Kenyan AA}}
\Big] \\
\text{s.t.} \quad & \sum_{j \in J} w_j x_{1j} \leq 400 \\
& \sum_{j \in J} w_j x_{2j} \leq 600 \\
& \sum_{j \in J} w_j x_{3j} \leq 500 \\
& \sum_{j \in J} w_j x_{4j} \leq 700 \\
& \sum_{j \in J} w_j x_{5j} \leq 450 \\
& \sum_{j \in J} w_j x_{6j} \leq 650 \\
& \sum_{j \in J} w_j x_{7j} \leq 550 \\
& \sum_{j \in J} w_j x_{8j} \leq 750 \\
& \sum_{j \in J} w_j x_{9j} \leq 480 \\
& \sum_{j \in J} w_j x_{10j} \leq 520 \\
& x_{ij} \in \mathbb{Z}_+, \quad \forall i \in I, j \in J
\end{align*}
\]

Where for each \( x_{ij} \), the coefficients \( v_j \) and \( w_j \) are as listed above, and all cabinets and products are indexed by their explicit IDs/names from the data.