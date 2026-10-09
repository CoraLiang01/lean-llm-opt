##### Decision Variables

- $x_{ij} \geq 0$: quantity shipped from warehouse $i \in I$ to store $j \in J$ (continuous).
- $y_i \in \{0,1\}$: whether warehouse $i$ is opened.

##### Parameters

- $I = \{1,2,3,4,5,6,7,8,9,10,11\}$ (warehouses)
- $J = \{1,2,3,4,5,6,7,8,9,10,11\}$ (stores)

- Warehouse opening costs $f_i$ and capacities $u_i$:

| Warehouse $i$ | $f_i$ (Opening Cost) | $u_i$ (Capacity) |
|:-------------:|:-------------------:|:----------------:|
| 1             | 3000                | 180              |
| 2             | 3200                | 160              |
| 3             | 3100                | 200              |
| 4             | 2800                | 150              |
| 5             | 3500                | 170              |
| 6             | 2700                | 190              |
| 7             | 2900                | 160              |
| 8             | 3050                | 175              |
| 9             | 3100                | 170              |
| 10            | 2200                | 180              |
| 11            | 2890                | 190              |

- Store demands $d_j$:

| Store $j$ | $d_j$ (Demand) |
|:---------:|:--------------:|
| 1         | 30             |
| 2         | 40             |
| 3         | 20             |
| 4         | 35             |
| 5         | 20             |
| 6         | 25             |
| 7         | 45             |
| 8         | 38             |
| 9         | 32             |
| 10        | 41             |
| 11        | 44             |

- Transportation costs $c_{ij}$ (from warehouse $i$ to store $j$):

| $c_{ij}$ | 1  | 2  | 3  | 4  | 5  | 6  | 7  | 8  | 9  | 10 | 11 |
|----------|----|----|----|----|----|----|----|----|----|----|----|
| 1        | 12 | 17 | 13 | 18 | 10 | 15 | 14 | 19 | 17 | 14 | 15 |
| 2        | 11 | 19 | 14 | 16 | 13 | 12 | 13 | 16 | 18 | 13 | 13 |
| 3        | 14 | 15 | 12 | 17 | 12 | 14 | 15 | 18 | 12 | 15 | 16 |
| 4        | 15 | 20 | 14 | 13 | 19 | 16 | 17 | 20 | 14 | 17 | 17 |
| 5        | 17 | 18 | 16 | 18 | 15 | 13 | 12 | 17 | 16 | 16 | 11 |
| 6        | 13 | 14 | 15 | 17 | 11 | 17 | 13 | 19 | 15 | 18 | 13 |
| 7        | 12 | 17 | 11 | 14 | 12 | 16 | 14 | 16 | 14 | 14 | 14 |
| 8        | 16 | 15 | 14 | 19 | 14 | 16 | 15 | 18 | 17 | 19 | 15 |
| 9        | 16 | 13 | 16 | 16 | 12 | 14 | 12 | 15 | 21 | 15 | 19 |
| 10       | 14 | 15 | 18 | 13 | 15 | 18 | 16 | 15 | 15 | 17 | 21 |
| 11       | 15 | 16 | 17 | 18 | 17 | 19 | 14 | 18 | 18 | 19 | 13 |

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i
\]

##### Constraints

1. **Demand satisfaction:**  
   For each store $j \in J$,
   \[
   \sum_{i \in I} x_{ij} = d_j
   \]

2. **Warehouse capacity:**  
   For each warehouse $i \in I$,
   \[
   \sum_{j \in J} x_{ij} \leq u_i y_i
   \]

3. **Variable domains:**  
   \[
   x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
   \]
   \[
   y_i \in \{0,1\} \quad \forall i \in I
   \]

---

###### Retrieved Information

```json
{
  "warehouses": {
    "1": {"Opening Cost": 3000, "Capacity": 180},
    "2": {"Opening Cost": 3200, "Capacity": 160},
    "3": {"Opening Cost": 3100, "Capacity": 200},
    "4": {"Opening Cost": 2800, "Capacity": 150},
    "5": {"Opening Cost": 3500, "Capacity": 170},
    "6": {"Opening Cost": 2700, "Capacity": 190},
    "7": {"Opening Cost": 2900, "Capacity": 160},
    "8": {"Opening Cost": 3050, "Capacity": 175},
    "9": {"Opening Cost": 3100, "Capacity": 170},
    "10": {"Opening Cost": 2200, "Capacity": 180},
    "11": {"Opening Cost": 2890, "Capacity": 190}
  },
  "stores": {
    "1": 30,
    "2": 40,
    "3": 20,
    "4": 35,
    "5": 20,
    "6": 25,
    "7": 45,
    "8": 38,
    "9": 32,
    "10": 41,
    "11": 44
  },
  "transportation_cost": [
    [12, 17, 13, 18, 10, 15, 14, 19, 17, 14, 15],
    [11, 19, 14, 16, 13, 12, 13, 16, 18, 13, 13],
    [14, 15, 12, 17, 12, 14, 15, 18, 12, 15, 16],
    [15, 20, 14, 13, 19, 16, 17, 20, 14, 17, 17],
    [17, 18, 16, 18, 15, 13, 12, 17, 16, 16, 11],
    [13, 14, 15, 17, 11, 17, 13, 19, 15, 18, 13],
    [12, 17, 11, 14, 12, 16, 14, 16, 14, 14, 14],
    [16, 15, 14, 19, 14, 16, 15, 18, 17, 19, 15],
    [16, 13, 16, 16, 12, 14, 12, 15, 21, 15, 19],
    [14, 15, 18, 13, 15, 18, 16, 15, 15, 17, 21],
    [15, 16, 17, 18, 17, 19, 14, 18, 18, 19, 13]
  ]
}
```
Where the $c_{ij}$ matrix is indexed by warehouse $i$ (rows 1–11) and store $j$ (columns 1–11).