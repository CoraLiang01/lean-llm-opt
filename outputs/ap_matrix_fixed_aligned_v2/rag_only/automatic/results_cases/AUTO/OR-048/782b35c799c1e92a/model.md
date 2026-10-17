Let:
- \( I = \{1,2,\ldots,15\} \) be the set of storage areas, indexed by StorageID as given in capacity.csv.
- \( J = \{\text{Window Unit}, \text{Portable Unit}, \text{Split System}, \text{Ductless System}, \text{Central AC}, \text{Hybrid AC}, \text{Geothermal AC}, \text{Smart AC}, \text{Evaporative Cooler}, \text{Package Unit}\} \) be the set of air conditioner types, indexed by ProductName as given in products.csv.

Parameters:
- For each storage area \( i \in I \), let \( \text{Capacity}_i \) be its capacity:
    - \( \text{Capacity}_1 = 1083 \)
    - \( \text{Capacity}_2 = 1840 \)
    - \( \text{Capacity}_3 = 770 \)
    - \( \text{Capacity}_4 = 1299 \)
    - \( \text{Capacity}_5 = 1259 \)
    - \( \text{Capacity}_6 = 543 \)
    - \( \text{Capacity}_7 = 1831 \)
    - \( \text{Capacity}_8 = 855 \)
    - \( \text{Capacity}_9 = 619 \)
    - \( \text{Capacity}_{10} = 637 \)
    - \( \text{Capacity}_{11} = 935 \)
    - \( \text{Capacity}_{12} = 626 \)
    - \( \text{Capacity}_{13} = 1457 \)
    - \( \text{Capacity}_{14} = 1198 \)
    - \( \text{Capacity}_{15} = 837 \)

- For each product \( j \in J \), let \( \text{Value}_j \) and \( \text{Weight}_j \) be as follows (from products.csv, in original order):

| ProductName           | Value | Weight |
|-----------------------|-------|--------|
| Window Unit           | 4811  | 114    |
| Portable Unit         | 1130  | 200    |
| Split System          | 1611  | 106    |
| Ductless System       | 3368  | 256    |
| Central AC            | 2135  | 268    |
| Hybrid AC             | 1046  | 185    |
| Geothermal AC         | 4030  | 299    |
| Smart AC              | 3761  | 131    |
| Evaporative Cooler    | 3523  | 139    |
| Package Unit          | 1701  | 105    |

Decision variables:
- For each storage area \( i \in I \) and product \( j \in J \), let \( x_{ij} \) be the number of units of product \( j \) to be placed in storage area \( i \).
- \( x_{ij} \) are nonnegative integers: \( x_{ij} \in \mathbb{Z}_{\geq 0} \).

Objective:
\[
\max \sum_{i \in I} \sum_{j \in J} \text{Value}_j \cdot x_{ij}
\]
That is,
\[
\max \sum_{i=1}^{15} \left(
4811\, x_{i,\text{Window Unit}} +
1130\, x_{i,\text{Portable Unit}} +
1611\, x_{i,\text{Split System}} +
3368\, x_{i,\text{Ductless System}} +
2135\, x_{i,\text{Central AC}} +
1046\, x_{i,\text{Hybrid AC}} +
4030\, x_{i,\text{Geothermal AC}} +
3761\, x_{i,\text{Smart AC}} +
3523\, x_{i,\text{Evaporative Cooler}} +
1701\, x_{i,\text{Package Unit}}
\right)
\]

Subject to, for each storage area \( i \in I \):
\[
\sum_{j \in J} \text{Weight}_j \cdot x_{ij} \leq \text{Capacity}_i
\]
That is, for each \( i \) (using the original order of products):
\[
114\, x_{i,\text{Window Unit}} +
200\, x_{i,\text{Portable Unit}} +
106\, x_{i,\text{Split System}} +
256\, x_{i,\text{Ductless System}} +
268\, x_{i,\text{Central AC}} +
185\, x_{i,\text{Hybrid AC}} +
299\, x_{i,\text{Geothermal AC}} +
131\, x_{i,\text{Smart AC}} +
139\, x_{i,\text{Evaporative Cooler}} +
105\, x_{i,\text{Package Unit}}
\leq \text{Capacity}_i
\]
where \( \text{Capacity}_i \) is as listed above for each StorageID.

Variable domains:
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in I,\, j \in J
\]

All data and indices are used as supplied, with no sorting or synthesis. This is a complete integer programming formulation for the described allocation problem.