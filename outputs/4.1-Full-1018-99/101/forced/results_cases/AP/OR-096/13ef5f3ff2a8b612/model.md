##### Sets and Indices

- Let $S$ be the set of schools: $S = \{\text{I}, \text{II}\}$
- Let $N$ be the set of neighborhoods: $N = \{\text{N01}, \text{N02}, \ldots, \text{N31}\}$
- Let $r \in \{\text{W}, \text{NW}\}$ denote race, where W = White, NW = Nonwhite.

##### Parameters

- $C_s$: Capacity of school $s \in S$
  - $C_{\text{I}} = 2028$
  - $C_{\text{II}} = 1560$
- $P_{n,r}$: Population of race $r$ in neighborhood $n \in N$
  - (see full table below)
- $d_{s,n}$: Distance in miles from school $s$ to neighborhood $n$
  - (see full table below)
- $P_{n,\text{W}}$, $P_{n,\text{NW}}$: White and nonwhite population in neighborhood $n$
- $P_{\text{total,W}} = \sum_{n \in N} P_{n,\text{W}}$
- $P_{\text{total,NW}} = \sum_{n \in N} P_{n,\text{NW}}$
- $P_{\text{total}} = P_{\text{total,W}} + P_{\text{total,NW}}$
- District white ratio: $r_{\text{district}} = \frac{P_{\text{total,W}}}{P_{\text{total}}} = 0.6$
- Allowed deviation: $\delta = 0.10$

##### Decision Variables

- $x_{s,n,r} \geq 0$: Number of students of race $r$ from neighborhood $n$ assigned to school $s$

##### Objective Function

Minimize total travel distance:
$$
\min \sum_{s \in S} \sum_{n \in N} \sum_{r \in \{\text{W}, \text{NW}\}} d_{s,n} \cdot x_{s,n,r}
$$

##### Constraints

1. **Neighborhood Assignment:**  
   All students from each neighborhood must be assigned to some school:
   $$
   \sum_{s \in S} x_{s,n,r} = P_{n,r} \quad \forall n \in N, \; r \in \{\text{W}, \text{NW}\}
   $$

2. **School Capacity:**  
   The total number of students assigned to each school cannot exceed its capacity:
   $$
   \sum_{n \in N} \sum_{r \in \{\text{W}, \text{NW}\}} x_{s,n,r} \leq C_s \quad \forall s \in S
   $$

3. **Racial Balance at Each School:**  
   The percentage of white students at each school must be within $[0.5, 0.7]$ (i.e., within 10 percentage points of the district ratio of 0.6):
   $$
   0.5 \leq \frac{\sum_{n \in N} x_{s,n,\text{W}}}{\sum_{n \in N} \sum_{r \in \{\text{W}, \text{NW}\}} x_{s,n,r}} \leq 0.7 \quad \forall s \in S
   $$
   (If the denominator is zero, the constraint is vacuously satisfied.)

4. **Nonnegativity:**
   $$
   x_{s,n,r} \geq 0 \quad \forall s \in S, n \in N, r \in \{\text{W}, \text{NW}\}
   $$

---

##### Retrieved Information

**School Capacities:**
```json
{
  "I": 2028,
  "II": 1560
}
```

**Neighborhood Populations:**
```json
{
  "N01": {"Population_White": 78, "Population_NonWhite": 22},
  "N02": {"Population_White": 57, "Population_NonWhite": 33},
  "N03": {"Population_White": 47, "Population_NonWhite": 63},
  "N04": {"Population_White": 78, "Population_NonWhite": 22},
  "N05": {"Population_White": 57, "Population_NonWhite": 33},
  "N06": {"Population_White": 47, "Population_NonWhite": 63},
  "N07": {"Population_White": 78, "Population_NonWhite": 22},
  "N08": {"Population_White": 57, "Population_NonWhite": 33},
  "N09": {"Population_White": 46, "Population_NonWhite": 64},
  "N10": {"Population_White": 77, "Population_NonWhite": 23},
  "N11": {"Population_White": 56, "Population_NonWhite": 34},
  "N12": {"Population_White": 46, "Population_NonWhite": 64},
  "N13": {"Population_White": 77, "Population_NonWhite": 23},
  "N14": {"Population_White": 56, "Population_NonWhite": 34},
  "N15": {"Population_White": 46, "Population_NonWhite": 64},
  "N16": {"Population_White": 77, "Population_NonWhite": 23},
  "N17": {"Population_White": 56, "Population_NonWhite": 34},
  "N18": {"Population_White": 46, "Population_NonWhite": 64},
  "N19": {"Population_White": 77, "Population_NonWhite": 23},
  "N20": {"Population_White": 56, "Population_NonWhite": 34},
  "N21": {"Population_White": 46, "Population_NonWhite": 64},
  "N22": {"Population_White": 77, "Population_NonWhite": 23},
  "N23": {"Population_White": 56, "Population_NonWhite": 34},
  "N24": {"Population_White": 46, "Population_NonWhite": 64},
  "N25": {"Population_White": 77, "Population_NonWhite": 23},
  "N26": {"Population_White": 56, "Population_NonWhite": 34},
  "N27": {"Population_White": 46, "Population_NonWhite": 64},
  "N28": {"Population_White": 77, "Population_NonWhite": 23},
  "N29": {"Population_White": 56, "Population_NonWhite": 34},
  "N30": {"Population_White": 46, "Population_NonWhite": 64},
  "N31": {"Population_White": 74, "Population_NonWhite": 46}
}
```

**Distances (miles):**
```json
{
  "I": {
    "N01": 1.25, "N02": 1.3, "N03": 1.35, "N04": 1.4, "N05": 1.45, "N06": 1.5, "N07": 1.55, "N08": 1.6, "N09": 1.65, "N10": 1.7, "N11": 1.75, "N12": 1.8, "N13": 1.85, "N14": 1.9, "N15": 1.95, "N16": 2.0, "N17": 3.08, "N18": 3.16, "N19": 3.24, "N20": 3.32, "N21": 3.4, "N22": 3.48, "N23": 3.56, "N24": 3.64, "N25": 3.72, "N26": 3.8, "N27": 3.88, "N28": 3.96, "N29": 4.04, "N30": 4.12, "N31": 4.2
  },
  "II": {
    "N01": 3.08, "N02": 3.16, "N03": 3.24, "N04": 3.32, "N05": 3.4, "N06": 3.48, "N07": 3.56, "N08": 3.64, "N09": 3.72, "N10": 3.8, "N11": 3.88, "N12": 3.96, "N13": 4.04, "N14": 4.12, "N15": 4.2, "N16": 4.28, "N17": 1.25, "N18": 1.3, "N19": 1.35, "N20": 1.4, "N21": 1.45, "N22": 1.5, "N23": 1.55, "N24": 1.6, "N25": 1.65, "N26": 1.7, "N27": 1.75, "N28": 1.8, "N29": 1.85, "N30": 1.9, "N31": 1.95
  }
}
```

**Sets:**
- Schools: $\{\text{I}, \text{II}\}$
- Neighborhoods: $\{\text{N01}, \ldots, \text{N31}\}$
- Races: $\{\text{W}, \text{NW}\}$

**Parameters:**
- $C_{\text{I}} = 2028$, $C_{\text{II}} = 1560$
- $P_{n,\text{W}}$, $P_{n,\text{NW}}$ as above
- $d_{s,n}$ as above

**District-wide white ratio:** $0.6$  
**Allowed deviation:** $\pm 0.10$ (i.e., $[0.5, 0.7]$)

---

This model assigns all white and nonwhite students from each neighborhood to schools, respects school capacities, enforces racial balance at each school, and minimizes total student travel distance.