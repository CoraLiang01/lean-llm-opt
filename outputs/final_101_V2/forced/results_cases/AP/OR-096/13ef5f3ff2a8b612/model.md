##### Sets and Indices

- Let $S$ be the set of schools: $S = \{\text{I}, \text{II}\}$
- Let $N$ be the set of neighborhoods: $N = \{\text{N01}, \text{N02}, \ldots, \text{N31}\}$
- Let $R$ be the set of races: $R = \{\text{White}, \text{NonWhite}\}$

Let $x_{s,n,r}$ be the number of students of race $r$ from neighborhood $n$ assigned to school $s$.

##### Parameters

- $C_s$: Capacity of school $s$ (from school_capacity.csv)
  - $C_{\text{I}} = 2028$
  - $C_{\text{II}} = 1560$
- $P_{n,\text{White}}$: White population in neighborhood $n$ (from neighborhoods_population.csv)
- $P_{n,\text{NonWhite}}$: Nonwhite population in neighborhood $n$ (from neighborhoods_population.csv)
- $d_{s,n}$: Distance in miles from school $s$ to neighborhood $n$ (from distance.csv)

###### Population Data

Neighborhood populations (by race):

| Neighborhood | $P_{n,\text{White}}$ | $P_{n,\text{NonWhite}}$ |
|--------------|----------------------|-------------------------|
| N01          | 78                   | 22                      |
| N02          | 57                   | 33                      |
| N03          | 47                   | 63                      |
| N04          | 78                   | 22                      |
| N05          | 57                   | 33                      |
| N06          | 47                   | 63                      |
| N07          | 78                   | 22                      |
| N08          | 57                   | 33                      |
| N09          | 46                   | 64                      |
| N10          | 77                   | 23                      |
| N11          | 56                   | 34                      |
| N12          | 46                   | 64                      |
| N13          | 77                   | 23                      |
| N14          | 56                   | 34                      |
| N15          | 46                   | 64                      |
| N16          | 77                   | 23                      |
| N17          | 56                   | 34                      |
| N18          | 46                   | 64                      |
| N19          | 77                   | 23                      |
| N20          | 56                   | 34                      |
| N21          | 46                   | 64                      |
| N22          | 77                   | 23                      |
| N23          | 56                   | 34                      |
| N24          | 46                   | 64                      |
| N25          | 77                   | 23                      |
| N26          | 56                   | 34                      |
| N27          | 46                   | 64                      |
| N28          | 77                   | 23                      |
| N29          | 56                   | 34                      |
| N30          | 46                   | 64                      |
| N31          | 74                   | 46                      |

###### Distance Data

Distances $d_{s,n}$ (in miles):

| School | N01  | N02  | N03  | N04  | N05  | N06  | N07  | N08  | N09  | N10  | N11  | N12  | N13  | N14  | N15  | N16  | N17  | N18  | N19  | N20  | N21  | N22  | N23  | N24  | N25  | N26  | N27  | N28  | N29  | N30  | N31  |
|--------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|
| I      | 1.25 | 1.3  | 1.35 | 1.4  | 1.45 | 1.5  | 1.55 | 1.6  | 1.65 | 1.7  | 1.75 | 1.8  | 1.85 | 1.9  | 1.95 | 2.0  | 3.08 | 3.16 | 3.24 | 3.32 | 3.4  | 3.48 | 3.56 | 3.64 | 3.72 | 3.8  | 3.88 | 3.96 | 4.04 | 4.12 | 4.2  |
| II     | 3.08 | 3.16 | 3.24 | 3.32 | 3.4  | 3.48 | 3.56 | 3.64 | 3.72 | 3.8  | 3.88 | 3.96 | 4.04 | 4.12 | 4.2  | 4.28 | 1.25 | 1.3  | 1.35 | 1.4  | 1.45 | 1.5  | 1.55 | 1.6  | 1.65 | 1.7  | 1.75 | 1.8  | 1.85 | 1.9  | 1.95 |

##### Objective Function

Minimize the total distance traveled by all students:

$$
\min \sum_{s \in S} \sum_{n \in N} \sum_{r \in R} d_{s,n} \cdot x_{s,n,r}
$$

##### Constraints

1. **Neighborhood Population Assignment**

All students of each race from each neighborhood must be assigned to some school:

$$
\sum_{s \in S} x_{s,n,r} = P_{n,r} \quad \forall n \in N, \; r \in R
$$

2. **School Capacity**

The total number of students assigned to each school cannot exceed its capacity:

$$
\sum_{n \in N} \sum_{r \in R} x_{s,n,r} \leq C_s \quad \forall s \in S
$$

3. **Racial Balance**

Let $T_s = \sum_{n \in N} \sum_{r \in R} x_{s,n,r}$ be the total number of students assigned to school $s$.

Let $W_s = \sum_{n \in N} x_{s,n,\text{White}}$ be the total number of white students assigned to school $s$.

The district-wide white percentage is 60%. Each school's white percentage must be within 10 percentage points of this, i.e., between 50% and 70%:

$$
0.5 \cdot T_s \leq W_s \leq 0.7 \cdot T_s \quad \forall s \in S
$$

4. **Nonnegativity and Integrality**

$$
x_{s,n,r} \geq 0 \quad \text{and integer} \quad \forall s \in S, \; n \in N, \; r \in R
$$

##### Retrieved Information

{
  "school_capacity": {
    "I": 2028,
    "II": 1560
  },
  "neighborhoods_population": {
    "N01": {"White": 78, "NonWhite": 22},
    "N02": {"White": 57, "NonWhite": 33},
    "N03": {"White": 47, "NonWhite": 63},
    "N04": {"White": 78, "NonWhite": 22},
    "N05": {"White": 57, "NonWhite": 33},
    "N06": {"White": 47, "NonWhite": 63},
    "N07": {"White": 78, "NonWhite": 22},
    "N08": {"White": 57, "NonWhite": 33},
    "N09": {"White": 46, "NonWhite": 64},
    "N10": {"White": 77, "NonWhite": 23},
    "N11": {"White": 56, "NonWhite": 34},
    "N12": {"White": 46, "NonWhite": 64},
    "N13": {"White": 77, "NonWhite": 23},
    "N14": {"White": 56, "NonWhite": 34},
    "N15": {"White": 46, "NonWhite": 64},
    "N16": {"White": 77, "NonWhite": 23},
    "N17": {"White": 56, "NonWhite": 34},
    "N18": {"White": 46, "NonWhite": 64},
    "N19": {"White": 77, "NonWhite": 23},
    "N20": {"White": 56, "NonWhite": 34},
    "N21": {"White": 46, "NonWhite": 64},
    "N22": {"White": 77, "NonWhite": 23},
    "N23": {"White": 56, "NonWhite": 34},
    "N24": {"White": 46, "NonWhite": 64},
    "N25": {"White": 77, "NonWhite": 23},
    "N26": {"White": 56, "NonWhite": 34},
    "N27": {"White": 46, "NonWhite": 64},
    "N28": {"White": 77, "NonWhite": 23},
    "N29": {"White": 56, "NonWhite": 34},
    "N30": {"White": 46, "NonWhite": 64},
    "N31": {"White": 74, "NonWhite": 46}
  },
  "distance": {
    "I": {
      "N01": 1.25, "N02": 1.3, "N03": 1.35, "N04": 1.4, "N05": 1.45, "N06": 1.5, "N07": 1.55, "N08": 1.6, "N09": 1.65, "N10": 1.7, "N11": 1.75, "N12": 1.8, "N13": 1.85, "N14": 1.9, "N15": 1.95, "N16": 2.0, "N17": 3.08, "N18": 3.16, "N19": 3.24, "N20": 3.32, "N21": 3.4, "N22": 3.48, "N23": 3.56, "N24": 3.64, "N25": 3.72, "N26": 3.8, "N27": 3.88, "N28": 3.96, "N29": 4.04, "N30": 4.12, "N31": 4.2
    },
    "II": {
      "N01": 3.08, "N02": 3.16, "N03": 3.24, "N04": 3.32, "N05": 3.4, "N06": 3.48, "N07": 3.56, "N08": 3.64, "N09": 3.72, "N10": 3.8, "N11": 3.88, "N12": 3.96, "N13": 4.04, "N14": 4.12, "N15": 4.2, "N16": 4.28, "N17": 1.25, "N18": 1.3, "N19": 1.35, "N20": 1.4, "N21": 1.45, "N22": 1.5, "N23": 1.55, "N24": 1.6, "N25": 1.65, "N26": 1.7, "N27": 1.75, "N28": 1.8, "N29": 1.85, "N30": 1.9, "N31": 1.95
    }
  }
}

##### Variable Domains

$x_{s,n,r} \in \mathbb{Z}_{\geq 0}$ for all $s \in S$, $n \in N$, $r \in R$