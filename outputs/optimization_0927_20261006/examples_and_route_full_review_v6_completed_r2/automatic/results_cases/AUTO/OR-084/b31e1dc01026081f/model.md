##### Parameters

- Let $T = \{1, 2, \ldots, 40\}$ be the set of tasks.
- Let $P = \{1, 2, 3\}$ be the set of CPUs.
- Let $f_1 = 1.33$ GHz, $f_2 = 2$ GHz, $f_3 = 2.66$ GHz be the frequencies of CPUs 1, 2, and 3, respectively.
- Let $b_i$ be the number of billions of instructions (BI) required by task $i$.

Task instruction counts from 18.csv:

\[
\begin{align*}
b_1 &= 1.1 \\
b_2 &= 2.1 \\
b_3 &= 3 \\
b_4 &= 1 \\
b_5 &= 0.7 \\
b_6 &= 5 \\
b_7 &= 3 \\
b_8 &= 3.5 \\
b_9 &= 4.4 \\
b_{10} &= 3.8 \\
b_{11} &= 3.5 \\
b_{12} &= 2.8 \\
b_{13} &= 4.1 \\
b_{14} &= 2.9 \\
b_{15} &= 5.4 \\
b_{16} &= 5.8 \\
b_{17} &= 2.6 \\
b_{18} &= 4.9 \\
b_{19} &= 3.4 \\
b_{20} &= 3.6 \\
b_{21} &= 5.6 \\
b_{22} &= 0.9 \\
b_{23} &= 1 \\
b_{24} &= 0.6 \\
b_{25} &= 5.1 \\
b_{26} &= 4.8 \\
b_{27} &= 5.3 \\
b_{28} &= 5.9 \\
b_{29} &= 4.9 \\
b_{30} &= 3 \\
b_{31} &= 4.8 \\
b_{32} &= 1.2 \\
b_{33} &= 4 \\
b_{34} &= 1.3 \\
b_{35} &= 5.7 \\
b_{36} &= 3.4 \\
b_{37} &= 2.8 \\
b_{38} &= 2 \\
b_{39} &= 4.8 \\
b_{40} &= 3 \\
\end{align*}
\]

##### Decision Variables

- $x_{i,j} = \begin{cases} 1 & \text{if task } i \text{ is assigned to CPU } j \\ 0 & \text{otherwise} \end{cases}$ for $i \in T$, $j \in P$
- $C_{\max}$: the makespan (completion time of the last finishing CPU)

##### Objective Function

\[
\min C_{\max}
\]

##### Constraints

1. **Each task is assigned to exactly one CPU:**

\[
\sum_{j=1}^3 x_{i,j} = 1 \quad \forall i \in T
\]

2. **Makespan definition:**

Let $S_j$ be the total processing time on CPU $j$:

\[
S_j = \sum_{i=1}^{40} \frac{b_i}{f_j} x_{i,j} \quad \forall j \in \{1,2,3\}
\]

\[
S_j \leq C_{\max} \quad \forall j \in \{1,2,3\}
\]

3. **Variable domains:**

\[
x_{i,j} \in \{0,1\} \quad \forall i \in T,\, j \in P
\]
\[
C_{\max} \geq 0
\]

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