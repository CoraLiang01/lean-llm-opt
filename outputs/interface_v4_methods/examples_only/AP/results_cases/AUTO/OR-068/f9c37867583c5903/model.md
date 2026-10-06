Let:
- M = {MA, MB, MC, MD, ME, MF} be the set of managers (indexed by i)
- P = {P1, P2, P3, P4, P5, P6} be the set of projects (indexed by j)
- c_{ij} be the cost of assigning manager i to project j, as given in the matrix below
- x_{ij} ∈ {0,1} be a binary variable, where x_{ij} = 1 if manager i is assigned to project j, 0 otherwise

Cost matrix C = [c_{ij}]:

|      | P1   | P2   | P3   | P4   | P5   | P6   |
|------|------|------|------|------|------|------|
| MA   | 2216 | 1911 | 1661 | 2122 | 1442 | 1442 |
| MB   | 1100 | 1271 | 2764 | 2557 | 1036 | 1036 |
| MC   | 2827 | 2784 | 2206 | 2216 | 2677 | 2677 |
| MD   | 2627 | 1273 | 2610 | 1957 | 1594 | 1594 |
| ME   | 3359 | 1003 | 2554 | 1706 | 2065 | 2065 |
| MF   | 1579 | 2289 | 2368 | 1922 | 2740 | 2740 |

Mathematical Model:

Decision variables:
x_{ij} = 
    1 if manager i ∈ M is assigned to project j ∈ P
    0 otherwise

Objective:
Minimize total assignment cost:
\[
\min \sum_{i \in M} \sum_{j \in P} c_{ij} x_{ij}
\]
That is,
\[
\min \Big(
2216x_{MA,P1} + 1911x_{MA,P2} + 1661x_{MA,P3} + 2122x_{MA,P4} + 1442x_{MA,P5} + 1442x_{MA,P6} \\
+ 1100x_{MB,P1} + 1271x_{MB,P2} + 2764x_{MB,P3} + 2557x_{MB,P4} + 1036x_{MB,P5} + 1036x_{MB,P6} \\
+ 2827x_{MC,P1} + 2784x_{MC,P2} + 2206x_{MC,P3} + 2216x_{MC,P4} + 2677x_{MC,P5} + 2677x_{MC,P6} \\
+ 2627x_{MD,P1} + 1273x_{MD,P2} + 2610x_{MD,P3} + 1957x_{MD,P4} + 1594x_{MD,P5} + 1594x_{MD,P6} \\
+ 3359x_{ME,P1} + 1003x_{ME,P2} + 2554x_{ME,P3} + 1706x_{ME,P4} + 2065x_{ME,P5} + 2065x_{ME,P6} \\
+ 1579x_{MF,P1} + 2289x_{MF,P2} + 2368x_{MF,P3} + 1922x_{MF,P4} + 2740x_{MF,P5} + 2740x_{MF,P6}
\Big)
\]

Subject to:

1. Each manager is assigned to exactly one project:
\[
\sum_{j \in P} x_{ij} = 1 \quad \forall i \in M
\]
That is,
\[
x_{MA,P1} + x_{MA,P2} + x_{MA,P3} + x_{MA,P4} + x_{MA,P5} + x_{MA,P6} = 1 \\
x_{MB,P1} + x_{MB,P2} + x_{MB,P3} + x_{MB,P4} + x_{MB,P5} + x_{MB,P6} = 1 \\
x_{MC,P1} + x_{MC,P2} + x_{MC,P3} + x_{MC,P4} + x_{MC,P5} + x_{MC,P6} = 1 \\
x_{MD,P1} + x_{MD,P2} + x_{MD,P3} + x_{MD,P4} + x_{MD,P5} + x_{MD,P6} = 1 \\
x_{ME,P1} + x_{ME,P2} + x_{ME,P3} + x_{ME,P4} + x_{ME,P5} + x_{ME,P6} = 1 \\
x_{MF,P1} + x_{MF,P2} + x_{MF,P3} + x_{MF,P4} + x_{MF,P5} + x_{MF,P6} = 1
\]

2. Each project is assigned to exactly one manager:
\[
\sum_{i \in M} x_{ij} = 1 \quad \forall j \in P
\]
That is,
\[
x_{MA,P1} + x_{MB,P1} + x_{MC,P1} + x_{MD,P1} + x_{ME,P1} + x_{MF,P1} = 1 \\
x_{MA,P2} + x_{MB,P2} + x_{MC,P2} + x_{MD,P2} + x_{ME,P2} + x_{MF,P2} = 1 \\
x_{MA,P3} + x_{MB,P3} + x_{MC,P3} + x_{MD,P3} + x_{ME,P3} + x_{MF,P3} = 1 \\
x_{MA,P4} + x_{MB,P4} + x_{MC,P4} + x_{MD,P4} + x_{ME,P4} + x_{MF,P4} = 1 \\
x_{MA,P5} + x_{MB,P5} + x_{MC,P5} + x_{MD,P5} + x_{ME,P5} + x_{MF,P5} = 1 \\
x_{MA,P6} + x_{MB,P6} + x_{MC,P6} + x_{MD,P6} + x_{ME,P6} + x_{MF,P6} = 1
\]

3. Binary variables:
\[
x_{ij} \in \{0,1\} \quad \forall i \in M, j \in P
\]

This model determines the minimum-cost one-to-one assignment of managers to projects, using the provided cost matrix.