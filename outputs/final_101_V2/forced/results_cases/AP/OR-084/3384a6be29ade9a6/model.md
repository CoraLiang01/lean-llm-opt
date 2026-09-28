##### Objective Function:

$\quad \min \; C_{\max}$

where $C_{\max}$ is the completion time of the last task (makespan).

##### Parameters:

- Let $T = \{1, 2, \ldots, 40\}$ be the set of tasks.
- Let $P = \{1, 2, 3\}$ be the set of CPUs.
- Let $f_1 = 1.33$ GHz, $f_2 = 2$ GHz, $f_3 = 2.66$ GHz be the frequencies of CPUs 1, 2, and 3, respectively.
- Let $b_i$ be the number of basic instructions (in billions) for task $i$.

Task instruction counts:

| Task | $b_i$ |
|------|-------|
| 1    | 1.1   |
| 2    | 2.1   |
| 3    | 3     |
| 4    | 1     |
| 5    | 0.7   |
| 6    | 5     |
| 7    | 3     |
| 8    | 3.5   |
| 9    | 4.4   |
| 10   | 3.8   |
| 11   | 3.5   |
| 12   | 2.8   |
| 13   | 4.1   |
| 14   | 2.9   |
| 15   | 5.4   |
| 16   | 5.8   |
| 17   | 2.6   |
| 18   | 4.9   |
| 19   | 3.4   |
| 20   | 3.6   |
| 21   | 5.6   |
| 22   | 0.9   |
| 23   | 1     |
| 24   | 0.6   |
| 25   | 5.1   |
| 26   | 4.8   |
| 27   | 5.3   |
| 28   | 5.9   |
| 29   | 4.9   |
| 30   | 3     |
| 31   | 4.8   |
| 32   | 1.2   |
| 33   | 4     |
| 34   | 1.3   |
| 35   | 5.7   |
| 36   | 3.4   |
| 37   | 2.8   |
| 38   | 2     |
| 39   | 4.8   |
| 40   | 3     |

##### Decision Variables:

- $x_{ip} \in \{0,1\}$: $x_{ip} = 1$ if task $i$ is assigned to CPU $p$, $0$ otherwise.
- $C_{\max} \geq 0$: the makespan.

##### Constraints:

1. **Each task is assigned to exactly one CPU:**

$\sum_{p=1}^3 x_{ip} = 1 \quad \forall i \in T$

2. **Makespan bounds the total processing time on each CPU:**

$\sum_{i=1}^{40} \frac{b_i}{f_p} x_{ip} \leq C_{\max} \quad \forall p \in P$

3. **Variable domains:**

$x_{ip} \in \{0,1\} \quad \forall i \in T, \; p \in P$

$C_{\max} \geq 0$

##### Retrieved Information

{
  "tasks": {
    "1": 1.1, "2": 2.1, "3": 3, "4": 1, "5": 0.7, "6": 5, "7": 3, "8": 3.5, "9": 4.4, "10": 3.8,
    "11": 3.5, "12": 2.8, "13": 4.1, "14": 2.9, "15": 5.4, "16": 5.8, "17": 2.6, "18": 4.9, "19": 3.4, "20": 3.6,
    "21": 5.6, "22": 0.9, "23": 1, "24": 0.6, "25": 5.1, "26": 4.8, "27": 5.3, "28": 5.9, "29": 4.9, "30": 3,
    "31": 4.8, "32": 1.2, "33": 4, "34": 1.3, "35": 5.7, "36": 3.4, "37": 2.8, "38": 2, "39": 4.8, "40": 3
  },
  "cpus": {
    "1": 1.33,
    "2": 2,
    "3": 2.66
  }
}