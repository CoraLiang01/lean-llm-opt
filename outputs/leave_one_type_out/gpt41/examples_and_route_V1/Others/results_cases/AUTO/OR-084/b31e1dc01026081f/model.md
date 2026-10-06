##### Parameters

- Let $T = \{1, 2, \ldots, 40\}$ be the set of tasks.
- Let $P = \{1, 2, 3\}$ be the set of CPUs.
- Let $b_i$ be the number of basic instructions (in billions) for task $i \in T$.
- Let $f_j$ be the frequency (in GHz) of CPU $j \in P$:
  - $f_1 = 1.33$
  - $f_2 = 2$
  - $f_3 = 2.66$

##### Variables

- $x_{ij} = \begin{cases} 1 & \text{if task } i \text{ is assigned to CPU } j \\ 0 & \text{otherwise} \end{cases}$
- $C_j$ = total processing time (in seconds) on CPU $j$
- $C_{\max}$ = makespan (completion time of the last task)

##### Objective Function

$\min C_{\max}$

##### Constraints

1. **Assignment Constraints:**

   Each task is assigned to exactly one CPU:
   $$
   \sum_{j=1}^3 x_{ij} = 1 \quad \forall i \in T
   $$

2. **CPU Load Calculation:**

   The total processing time on each CPU is the sum of the processing times of its assigned tasks:
   $$
   C_j = \sum_{i=1}^{40} \frac{b_i}{f_j} x_{ij} \quad \forall j \in P
   $$

3. **Makespan Definition:**

   The makespan is at least as large as the load on any CPU:
   $$
   C_{\max} \geq C_j \quad \forall j \in P
   $$

4. **Variable Domains:**

   $$
   x_{ij} \in \{0,1\} \quad \forall i \in T, \forall j \in P
   $$
   $$
   C_j \geq 0 \quad \forall j \in P
   $$
   $$
   C_{\max} \geq 0
   $$

##### Retrieved Information

{
  "tasks": {
    "1": 1.1,
    "2": 2.1,
    "3": 3,
    "4": 1,
    "5": 0.7,
    "6": 5,
    "7": 3,
    "8": 3.5,
    "9": 4.4,
    "10": 3.8,
    "11": 3.5,
    "12": 2.8,
    "13": 4.1,
    "14": 2.9,
    "15": 5.4,
    "16": 5.8,
    "17": 2.6,
    "18": 4.9,
    "19": 3.4,
    "20": 3.6,
    "21": 5.6,
    "22": 0.9,
    "23": 1,
    "24": 0.6,
    "25": 5.1,
    "26": 4.8,
    "27": 5.3,
    "28": 5.9,
    "29": 4.9,
    "30": 3,
    "31": 4.8,
    "32": 1.2,
    "33": 4,
    "34": 1.3,
    "35": 5.7,
    "36": 3.4,
    "37": 2.8,
    "38": 2,
    "39": 4.8,
    "40": 3
  },
  "cpu_frequencies": {
    "1": 1.33,
    "2": 2,
    "3": 2.66
  }
}