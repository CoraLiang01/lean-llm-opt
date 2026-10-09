**Mathematical Optimization Model**

Let  
- $x_{ij}$ = number of units of air conditioner type $j$ to be placed in storage area $i$  
  ($i \in \{\text{1},\ldots,\text{15}\}$, $j \in \{\text{Window Unit}, \text{Portable Unit}, \text{Split System}, \text{Ductless System}, \text{Central AC}, \text{Hybrid AC}, \text{Geothermal AC}, \text{Smart AC}, \text{Evaporative Cooler}, \text{Package Unit}\}$)

**Parameters:**

- Storage areas and capacities:

| StorageID | Capacity |
|-----------|----------|
| 1         | 1083     |
| 2         | 1840     |
| 3         | 770      |
| 4         | 1299     |
| 5         | 1259     |
| 6         | 543      |
| 7         | 1831     |
| 8         | 855      |
| 9         | 619      |
| 10        | 637      |
| 11        | 935      |
| 12        | 626      |
| 13        | 1457     |
| 14        | 1198     |
| 15        | 837      |

- Air conditioner types, values, and weights:

| ProductName         | Value | Weight |
|---------------------|-------|--------|
| Window Unit         | 4811  | 114    |
| Portable Unit       | 1130  | 200    |
| Split System        | 1611  | 106    |
| Ductless System     | 3368  | 256    |
| Central AC          | 2135  | 268    |
| Hybrid AC           | 1046  | 185    |
| Geothermal AC       | 4030  | 299    |
| Smart AC            | 3761  | 131    |
| Evaporative Cooler  | 3523  | 139    |
| Package Unit        | 1701  | 105    |

---

### Objective

$$
\max \sum_{i=1}^{15} \Bigg( 
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
\Bigg)
$$

---

### Constraints

For each storage area $i$ (where $i$ is the StorageID from 1 to 15):

$$
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
$$

where $\text{Capacity}_i$ is as given in the table above for each $i$.

That is, for each $i$:

- $i=1$: $\leq 1083$
- $i=2$: $\leq 1840$
- $i=3$: $\leq 770$
- $i=4$: $\leq 1299$
- $i=5$: $\leq 1259$
- $i=6$: $\leq 543$
- $i=7$: $\leq 1831$
- $i=8$: $\leq 855$
- $i=9$: $\leq 619$
- $i=10$: $\leq 637$
- $i=11$: $\leq 935$
- $i=12$: $\leq 626$
- $i=13$: $\leq 1457$
- $i=14$: $\leq 1198$
- $i=15$: $\leq 837$

---

### Variable Domains

$$
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in \{1,\ldots,15\},\ j \in \{\text{Window Unit}, \text{Portable Unit}, \text{Split System}, \text{Ductless System}, \text{Central AC}, \text{Hybrid AC}, \text{Geothermal AC}, \text{Smart AC}, \text{Evaporative Cooler}, \text{Package Unit}\}
$$

---

**Summary:**  
Maximize the total value of air conditioners allocated to storage areas, subject to each area's capacity, using integer variables for the number of units of each type in each area, with all coefficients and identifiers as above.