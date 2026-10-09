Let $x_k$ be the number of units of radio model HiFi-$k$ ($k=1,\ldots,101$) to produce per day. All $x_k$ are nonnegative integers.

Let $a_{jk}$ be the processing time (in minutes) required for one unit of model $k$ at workstation $j$ ($j=1,2,3$), as given below.

Let $C_j$ be the effective daily capacity (in minutes) of workstation $j$ after maintenance.

Define the idle time at workstation $j$ as $I_j = C_j - \sum_{k=1}^{101} a_{jk} x_k$.

**Parameters from the data:**

- For workstation 1: $C_1 = 1440 \times (1 - 0.10) = 1296$ minutes
- For workstation 2: $C_2 = 1440 \times (1 - 0.14) = 1238.4$ minutes
- For workstation 3: $C_3 = 1440 \times (1 - 0.12) = 1267.2$ minutes

- The $a_{jk}$ coefficients are as follows (from the CSV, in source order):

| $k$ | Model         | $a_{1k}$ | $a_{2k}$ | $a_{3k}$ |
|-----|--------------|----------|----------|----------|
| 1   | HiFi1        | 6        | 5        | 4        |
| 2   | HiFi2        | 4        | 5        | 6        |
| 3   | HiFi3        | 6        | 5        | 5        |
| 4   | HiFi4        | 7        | 1        | 2        |
| 5   | HiFi5        | 6        | 7        | 6        |
| 6   | HiFi6        | 6        | 8        | 5        |
| 7   | HiFi7        | 8        | 7        | 3        |
| 8   | HiFi8        | 9        | 5        | 3        |
| 9   | HiFi9        | 6        | 6        | 4        |
| 10  | HiFi10       | 7        | 8        | 8        |
| 11  | HiFi11       | 1        | 9        | 6        |
| 12  | HiFi12       | 2        | 9        | 3        |
| 13  | HiFi13       | 4        | 2        | 3        |
| 14  | HiFi14       | 7        | 6        | 3        |
| 15  | HiFi15       | 3        | 9        | 7        |
| 16  | HiFi16       | 8        | 4        | 8        |
| 17  | HiFi17       | 3        | 1        | 3        |
| 18  | HiFi18       | 2        | 2        | 8        |
| 19  | HiFi19       | 4        | 9        | 1        |
| 20  | HiFi20       | 5        | 3        | 5        |
| 21  | HiFi21       | 8        | 8        | 3        |
| 22  | HiFi22       | 3        | 5        | 8        |
| 23  | HiFi23       | 2        | 9        | 5        |
| 24  | HiFi24       | 3        | 5        | 8        |
| 25  | HiFi25       | 9        | 8        | 4        |
| 26  | HiFi26       | 7        | 7        | 8        |
| 27  | HiFi27       | 3        | 1        | 6        |
| 28  | HiFi28       | 5        | 1        | 7        |
| 29  | HiFi29       | 7        | 9        | 9        |
| 30  | HiFi30       | 6        | 7        | 5        |
| 31  | HiFi31       | 2        | 1        | 3        |
| 32  | HiFi32       | 1        | 9        | 6        |
| 33  | HiFi33       | 5        | 6        | 3        |
| 34  | HiFi34       | 6        | 4        | 3        |
| 35  | HiFi35       | 5        | 7        | 3        |
| 36  | HiFi36       | 1        | 4        | 8        |
| 37  | HiFi37       | 7        | 8        | 4        |
| 38  | HiFi38       | 9        | 6        | 6        |
| 39  | HiFi39       | 8        | 5        | 3        |
| 40  | HiFi40       | 3        | 3        | 8        |
| 41  | HiFi41       | 3        | 6        | 3        |
| 42  | HiFi42       | 8        | 7        | 7        |
| 43  | HiFi43       | 2        | 6        | 5        |
| 44  | HiFi44       | 3        | 2        | 3        |
| 45  | HiFi45       | 3        | 1        | 1        |
| 46  | HiFi46       | 8        | 1        | 8        |
| 47  | HiFi47       | 9        | 3        | 9        |
| 48  | HiFi48       | 2        | 8        | 6        |
| 49  | HiFi49       | 3        | 4        | 6        |
| 50  | HiFi50       | 4        | 3        | 4        |
| 51  | HiFi51       | 2        | 6        | 7        |
| 52  | HiFi52       | 9        | 9        | 1        |
| 53  | HiFi53       | 2        | 8        | 9        |
| 54  | HiFi54       | 1        | 7        | 9        |
| 55  | HiFi55       | 8        | 2        | 3        |
| 56  | HiFi56       | 8        | 2        | 9        |
| 57  | HiFi57       | 4        | 5        | 6        |
| 58  | HiFi58       | 4        | 4        | 5        |
| 59  | HiFi59       | 6        | 3        | 7        |
| 60  | HiFi60       | 1        | 8        | 8        |
| 61  | HiFi61       | 6        | 8        | 9        |
| 62  | HiFi62       | 5        | 6        | 9        |
| 63  | HiFi63       | 3        | 6        | 8        |
| 64  | HiFi64       | 5        | 3        | 5        |
| 65  | HiFi65       | 1        | 1        | 4        |
| 66  | HiFi66       | 6        | 6        | 4        |
| 67  | HiFi67       | 6        | 2        | 3        |
| 68  | HiFi68       | 5        | 6        | 3        |
| 69  | HiFi69       | 3        | 1        | 8        |
| 70  | HiFi70       | 4        | 3        | 8        |
| 71  | HiFi71       | 3        | 7        | 2        |
| 72  | HiFi72       | 8        | 1        | 4        |
| 73  | HiFi73       | 1        | 1        | 9        |
| 74  | HiFi74       | 2        | 2        | 6        |
| 75  | HiFi75       | 3        | 8        | 7        |
| 76  | HiFi76       | 2        | 7        | 6        |
| 77  | HiFi77       | 8        | 8        | 7        |
| 78  | HiFi78       | 4        | 8        | 3        |
| 79  | HiFi79       | 4        | 7        | 1        |
| 80  | HiFi80       | 2        | 5        | 7        |
| 81  | HiFi81       | 7        | 2        | 6        |
| 82  | HiFi82       | 5        | 5        | 4        |
| 83  | HiFi83       | 1        | 6        | 3        |
| 84  | HiFi84       | 6        | 2        | 5        |
| 85  | HiFi85       | 4        | 3        | 7        |
| 86  | HiFi86       | 1        | 2        | 6        |
| 87  | HiFi87       | 3        | 3        | 3        |
| 88  | HiFi88       | 8        | 8        | 5        |
| 89  | HiFi89       | 3        | 4        | 2        |
| 90  | HiFi90       | 3        | 9        | 2        |
| 91  | HiFi91       | 3        | 6        | 9        |
| 92  | HiFi92       | 3        | 1        | 3        |
| 93  | HiFi93       | 6        | 4        | 6        |
| 94  | HiFi94       | 7        | 8        | 9        |
| 95  | HiFi95       | 6        | 8        | 7        |
| 96  | HiFi96       | 2        | 6        | 2        |
| 97  | HiFi97       | 1        | 8        | 4        |
| 98  | HiFi98       | 8        | 5        | 5        |
| 99  | HiFi99       | 9        | 5        | 8        |
| 100 | HiFi100      | 7        | 8        | 1        |
| 101 | HiFi101      | 9        | 3        | 6        |

**Mathematical Model:**

Minimize total idle time:
$$
\min \left[ (C_1 - \sum_{k=1}^{101} a_{1k} x_k) + (C_2 - \sum_{k=1}^{101} a_{2k} x_k) + (C_3 - \sum_{k=1}^{101} a_{3k} x_k) \right]
$$

Equivalently,
$$
\min \left[ (C_1 + C_2 + C_3) - \sum_{j=1}^3 \sum_{k=1}^{101} a_{jk} x_k \right]
$$

Since $C_1 + C_2 + C_3$ is constant, this is equivalent to:
$$
\max \sum_{j=1}^3 \sum_{k=1}^{101} a_{jk} x_k
$$

subject to:

For each workstation $j=1,2,3$:
$$
\sum_{k=1}^{101} a_{jk} x_k \leq C_j
$$

and
$$
x_k \in \mathbb{Z}_{\geq 0}, \quad \forall k=1,\ldots,101
$$

**Where:**

- $a_{jk}$ are the processing times from the table above (from the CSV, in source order).
- $C_1 = 1296$, $C_2 = 1238.4$, $C_3 = 1267.2$.

**Decision variables:**

- $x_k$: number of units of HiFi-$k$ to produce per day, $x_k \in \mathbb{Z}_{\geq 0}$.

**Complete Formulation:**

$$
\begin{align*}
\min \quad & \sum_{j=1}^3 \left[ C_j - \sum_{k=1}^{101} a_{jk} x_k \right] \\
\text{s.t.} \quad & \sum_{k=1}^{101} a_{jk} x_k \leq C_j, \quad j=1,2,3 \\
& x_k \in \mathbb{Z}_{\geq 0}, \quad k=1,\ldots,101
\end{align*}
$$

with all coefficients and capacities as above, in the original source order.