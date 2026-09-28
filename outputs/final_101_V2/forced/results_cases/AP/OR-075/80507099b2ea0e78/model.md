##### Decision Variables

Let $x_i$ be a binary variable for each project $i \in \{1,2,\ldots,110\}$:
$$
x_i = \begin{cases}
1 & \text{if project } i \text{ is selected} \\
0 & \text{otherwise}
\end{cases}
$$

##### Parameters

Let $c_i$ be the capital investment required for project $i$ (in k$), and $n_i$ be the expected NPV for project $i$ (in k$), as given below:

| $i$ | Project Name                        | $c_i$ | $n_i$ |
|-----|-------------------------------------|-------|-------|
| 1   | Infrastructure Upgrade              | 50    | 60    |
| 2   | New Product Line A                  | 40    | 50    |
| 3   | Marketing Campaign X                | 30    | 45    |
| 4   | R&D Initiative Alpha                | 25    | 35    |
| 5   | Staff Training Program              | 20    | 28    |
| 6   | System Automation                   | 65    | 75    |
| 7   | Global Expansion Pilot              | 80    | 100   |
| 8   | Green Energy Switch                 | 15    | 20    |
| 9   | Warehouse Optimization              | 48    | 62    |
| 10  | Customer Experience Platform        | 55    | 70    |
| 11  | Project 011 - Strategy Focus        | 87    | 117   |
| 12  | Project 012 - Strategy Focus        | 82    | 109   |
| 13  | Project 013 - Strategy Focus        | 123   | 172   |
| 14  | Project 014 - Strategy Focus        | 137   | 180   |
| 15  | Project 015 - Strategy Focus        | 43    | 62    |
| 16  | Project 016 - Strategy Focus        | 110   | 147   |
| 17  | Project 017 - Strategy Focus        | 94    | 141   |
| 18  | Project 018 - Strategy Focus        | 101   | 151   |
| 19  | Project 019 - Strategy Focus        | 54    | 75    |
| 20  | Project 020 - Strategy Focus        | 143   | 185   |
| 21  | Project 021 - Strategy Focus        | 37    | 51    |
| 22  | Project 022 - Strategy Focus        | 46    | 59    |
| 23  | Project 023 - Strategy Focus        | 81    | 105   |
| 24  | Project 024 - Strategy Focus        | 100   | 123   |
| 25  | Project 025 - Strategy Focus        | 80    | 104   |
| 26  | Project 026 - Strategy Focus        | 102   | 153   |
| 27  | Project 027 - Strategy Focus        | 109   | 163   |
| 28  | Project 028 - Strategy Focus        | 79    | 114   |
| 29  | Project 029 - Strategy Focus        | 100   | 125   |
| 30  | Project 030 - Strategy Focus        | 119   | 160   |
| 31  | Project 031 - Strategy Focus        | 139   | 166   |
| 32  | Project 032 - Strategy Focus        | 11    | 15    |
| 33  | Project 033 - Strategy Focus        | 144   | 184   |
| 34  | Project 034 - Strategy Focus        | 56    | 86    |
| 35  | Project 035 - Strategy Focus        | 59    | 85    |
| 36  | Project 036 - Strategy Focus        | 73    | 109   |
| 37  | Project 037 - Strategy Focus        | 136   | 187   |
| 38  | Project 038 - Strategy Focus        | 85    | 115   |
| 39  | Project 039 - Strategy Focus        | 111   | 155   |
| 40  | Project 040 - Strategy Focus        | 103   | 137   |
| 41  | Project 041 - Strategy Focus        | 146   | 204   |
| 42  | Project 042 - Strategy Focus        | 82    | 98    |
| 43  | Project 043 - Strategy Focus        | 144   | 189   |
| 44  | Project 044 - Strategy Focus        | 143   | 178   |
| 45  | Project 045 - Strategy Focus        | 85    | 130   |
| 46  | Project 046 - Strategy Focus        | 121   | 148   |
| 47  | Project 047 - Strategy Focus        | 147   | 205   |
| 48  | Project 048 - Strategy Focus        | 92    | 128   |
| 49  | Project 049 - Strategy Focus        | 115   | 165   |
| 50  | Project 050 - Strategy Focus        | 13    | 18    |
| 51  | Project 051 - Strategy Focus        | 145   | 203   |
| 52  | Project 052 - Strategy Focus        | 48    | 72    |
| 53  | Project 053 - Strategy Focus        | 74    | 92    |
| 54  | Project 054 - Strategy Focus        | 135   | 167   |
| 55  | Project 055 - Strategy Focus        | 28    | 40    |
| 56  | Project 056 - Strategy Focus        | 74    | 89    |
| 57  | Project 057 - Strategy Focus        | 89    | 111   |
| 58  | Project 058 - Strategy Focus        | 24    | 38    |
| 59  | Project 059 - Strategy Focus        | 141   | 190   |
| 60  | Project 060 - Strategy Focus        | 27    | 33    |
| 61  | Project 061 - Strategy Focus        | 106   | 148   |
| 62  | Project 062 - Strategy Focus        | 37    | 44    |
| 63  | Project 063 - Strategy Focus        | 114   | 167   |
| 64  | Project 064 - Strategy Focus        | 11    | 14    |
| 65  | Project 065 - Strategy Focus        | 113   | 153   |
| 66  | Project 066 - Strategy Focus        | 61    | 91    |
| 67  | Project 067 - Strategy Focus        | 62    | 94    |
| 68  | Project 068 - Strategy Focus        | 145   | 188   |
| 69  | Project 069 - Strategy Focus        | 148   | 177   |
| 70  | Project 070 - Strategy Focus        | 30    | 36    |
| 71  | Project 071 - Strategy Focus        | 134   | 179   |
| 72  | Project 072 - Strategy Focus        | 97    | 143   |
| 73  | Project 073 - Strategy Focus        | 103   | 123   |
| 74  | Project 074 - Strategy Focus        | 102   | 126   |
| 75  | Project 075 - Strategy Focus        | 141   | 173   |
| 76  | Project 076 - Strategy Focus        | 109   | 152   |
| 77  | Project 077 - Strategy Focus        | 118   | 170   |
| 78  | Project 078 - Strategy Focus        | 93    | 148   |
| 79  | Project 079 - Strategy Focus        | 95    | 142   |
| 80  | Project 080 - Strategy Focus        | 92    | 118   |
| 81  | Project 081 - Strategy Focus        | 71    | 104   |
| 82  | Project 082 - Strategy Focus        | 127   | 159   |
| 83  | Project 083 - Strategy Focus        | 126   | 163   |
| 84  | Project 084 - Strategy Focus        | 123   | 165   |
| 85  | Project 085 - Strategy Focus        | 137   | 164   |
| 86  | Project 086 - Strategy Focus        | 124   | 149   |
| 87  | Project 087 - Strategy Focus        | 103   | 144   |
| 88  | Project 088 - Strategy Focus        | 119   | 166   |
| 89  | Project 089 - Strategy Focus        | 87    | 105   |
| 90  | Project 090 - Strategy Focus        | 87    | 130   |
| 91  | Project 091 - Strategy Focus        | 92    | 138   |
| 92  | Project 092 - Strategy Focus        | 69    | 96    |
| 93  | Project 093 - Strategy Focus        | 149   | 178   |
| 94  | Project 094 - Strategy Focus        | 146   | 193   |
| 95  | Project 095 - Strategy Focus        | 47    | 68    |
| 96  | Project 096 - Strategy Focus        | 12    | 18    |
| 97  | Project 097 - Strategy Focus        | 101   | 131   |
| 98  | Project 098 - Strategy Focus        | 69    | 103   |
| 99  | Project 099 - Strategy Focus        | 46    | 65    |
| 100 | Project 100 - Strategy Focus        | 79    | 108   |
| 101 | Project 101 - Strategy Focus        | 93    | 130   |
| 102 | Project 102 - Strategy Focus        | 49    | 60    |
| 103 | Project 103 - Strategy Focus        | 110   | 157   |
| 104 | Project 104 - Strategy Focus        | 133   | 184   |
| 105 | Project 105 - Strategy Focus        | 92    | 130   |
| 106 | Project 106 - Strategy Focus        | 106   | 128   |
| 107 | Project 107 - Strategy Focus        | 144   | 201   |
| 108 | Project 108 - Strategy Focus        | 124   | 173   |
| 109 | Project 109 - Strategy Focus        | 148   | 189   |
| 110 | Project 110 - Strategy Focus        | 127   | 177   |

##### Objective Function

$$
\max \sum_{i=1}^{110} n_i x_i
$$

##### Constraints

1. **Budget Constraint:**
   $$
   \sum_{i=1}^{110} c_i x_i \leq 1000
   $$

2. **Mutually Exclusive Constraint (Projects 4 & 7):**
   $$
   x_4 + x_7 \leq 1
   $$

3. **Pre-requisite Constraint (Project 6 requires 1):**
   $$
   x_6 \leq x_1
   $$

4. **Contingent Constraint (Project 10 requires 5):**
   $$
   x_{10} \leq x_5
   $$

5. **Binary Variables:**
   $$
   x_i \in \{0,1\} \quad \forall i \in \{1,2,\ldots,110\}
   $$

##### Retrieved Information

{
  "projects": [
    {"id": 1, "name": "Infrastructure Upgrade", "capital": 50, "npv": 60},
    {"id": 2, "name": "New Product Line A", "capital": 40, "npv": 50},
    {"id": 3, "name": "Marketing Campaign X", "capital": 30, "npv": 45},
    {"id": 4, "name": "R&D Initiative Alpha", "capital": 25, "npv": 35},
    {"id": 5, "name": "Staff Training Program", "capital": 20, "npv": 28},
    {"id": 6, "name": "System Automation", "capital": 65, "npv": 75},
    {"id": 7, "name": "Global Expansion Pilot", "capital": 80, "npv": 100},
    {"id": 8, "name": "Green Energy Switch", "capital": 15, "npv": 20},
    {"id": 9, "name": "Warehouse Optimization", "capital": 48, "npv": 62},
    {"id": 10, "name": "Customer Experience Platform", "capital": 55, "npv": 70},
    {"id": 11, "name": "Project 011 - Strategy Focus", "capital": 87, "npv": 117},
    {"id": 12, "name": "Project 012 - Strategy Focus", "capital": 82, "npv": 109},
    {"id": 13, "name": "Project 013 - Strategy Focus", "capital": 123, "npv": 172},
    {"id": 14, "name": "Project 014 - Strategy Focus", "capital": 137, "npv": 180},
    {"id": 15, "name": "Project 015 - Strategy Focus", "capital": 43, "npv": 62},
    {"id": 16, "name": "Project 016 - Strategy Focus", "capital": 110, "npv": 147},
    {"id": 17, "name": "Project 017 - Strategy Focus", "capital": 94, "npv": 141},
    {"id": 18, "name": "Project 018 - Strategy Focus", "capital": 101, "npv": 151},
    {"id": 19, "name": "Project 019 - Strategy Focus", "capital": 54, "npv": 75},
    {"id": 20, "name": "Project 020 - Strategy Focus", "capital": 143, "npv": 185},
    {"id": 21, "name": "Project 021 - Strategy Focus", "capital": 37, "npv": 51},
    {"id": 22, "name": "Project 022 - Strategy Focus", "capital": 46, "npv": 59},
    {"id": 23, "name": "Project 023 - Strategy Focus", "capital": 81, "npv": 105},
    {"id": 24, "name": "Project 024 - Strategy Focus", "capital": 100, "npv": 123},
    {"id": 25, "name": "Project 025 - Strategy Focus", "capital": 80, "npv": 104},
    {"id": 26, "name": "Project 026 - Strategy Focus", "capital": 102, "npv": 153},
    {"id": 27, "name": "Project 027 - Strategy Focus", "capital": 109, "npv": 163},
    {"id": 28, "name": "Project 028 - Strategy Focus", "capital": 79, "npv": 114},
    {"id": 29, "name": "Project 029 - Strategy Focus", "capital": 100, "npv": 125},
    {"id": 30, "name": "Project 030 - Strategy Focus", "capital": 119, "npv": 160},
    {"id": 31, "name": "Project 031 - Strategy Focus", "capital": 139, "npv": 166},
    {"id": 32, "name": "Project 032 - Strategy Focus", "capital": 11, "npv": 15},
    {"id": 33, "name": "Project 033 - Strategy Focus", "capital": 144, "npv": 184},
    {"id": 34, "name": "Project 034 - Strategy Focus", "capital": 56, "npv": 86},
    {"id": 35, "name": "Project 035 - Strategy Focus", "capital": 59, "npv": 85},
    {"id": 36, "name": "Project 036 - Strategy Focus", "capital": 73, "npv": 109},
    {"id": 37, "name": "Project 037 - Strategy Focus", "capital": 136, "npv": 187},
    {"id": 38, "name": "Project 038 - Strategy Focus", "capital": 85, "npv": 115},
    {"id": 39, "name": "Project 039 - Strategy Focus", "capital": 111, "npv": 155},
    {"id": 40, "name": "Project 040 - Strategy Focus", "capital": 103, "npv": 137},
    {"id": 41, "name": "Project 041 - Strategy Focus", "capital": 146, "npv": 204},
    {"id": 42, "name": "Project 042 - Strategy Focus", "capital": 82, "npv": 98},
    {"id": 43, "name": "Project 043 - Strategy Focus", "capital": 144, "npv": 189},
    {"id": 44, "name": "Project 044 - Strategy Focus", "capital": 143, "npv": 178},
    {"id": 45, "name": "Project 045 - Strategy Focus", "capital": 85, "npv": 130},
    {"id": 46, "name": "Project 046 - Strategy Focus", "capital": 121, "npv": 148},
    {"id": 47, "name": "Project 047 - Strategy Focus", "capital": 147, "npv": 205},
    {"id": 48, "name": "Project 048 - Strategy Focus", "capital": 92, "npv": 128},
    {"id": 49, "name": "Project 049 - Strategy Focus", "capital": 115, "npv": 165},
    {"id": 50, "name": "Project 050 - Strategy Focus", "capital": 13, "npv": 18},
    {"id": 51, "name": "Project 051 - Strategy Focus", "capital": 145, "npv": 203},
    {"id": 52, "name": "Project 052 - Strategy Focus", "capital": 48, "npv": 72},
    {"id": 53, "name": "Project 053 - Strategy Focus", "capital": 74, "npv": 92},
    {"id": 54, "name": "Project 054 - Strategy Focus", "capital": 135, "npv": 167},
    {"id": 55, "name": "Project 055 - Strategy Focus", "capital": 28, "npv": 40},
    {"id": 56, "name": "Project 056 - Strategy Focus", "capital": 74, "npv": 89},
    {"id": 57, "name": "Project 057 - Strategy Focus", "capital": 89, "npv": 111},
    {"id": 58, "name": "Project 058 - Strategy Focus", "capital": 24, "npv": 38},
    {"id": 59, "name": "Project 059 - Strategy Focus", "capital": 141, "npv": 190},
    {"id": 60, "name": "Project 060 - Strategy Focus", "capital": 27, "npv": 33},
    {"id": 61, "name": "Project 061 - Strategy Focus", "capital": 106, "npv": 148},
    {"id": 62, "name": "Project 062 - Strategy Focus", "capital": 37, "npv": 44},
    {"id": 63, "name": "Project 063 - Strategy Focus", "capital": 114, "npv": 167},
    {"id": 64, "name": "Project 064 - Strategy Focus", "capital": 11, "npv": 14},
    {"id": 65, "name": "Project 065 - Strategy Focus", "capital": 113, "npv": 153},
    {"id": 66, "name": "Project 066 - Strategy Focus", "capital": 61, "npv": 91},
    {"id": 67, "name": "Project 067 - Strategy Focus", "capital": 62, "npv": 94},
    {"id": 68, "name": "Project 068 - Strategy Focus", "capital": 145, "npv": 188},
    {"id": 69, "name": "Project 069 - Strategy Focus", "capital": 148, "npv": 177},
    {"id": 70, "name": "Project 070 - Strategy Focus", "capital": 30, "npv": 36},
    {"id": 71, "name": "Project 071 - Strategy Focus", "capital": 134, "npv": 179},
    {"id": 72, "name": "Project 072 - Strategy Focus", "capital": 97, "npv": 143},
    {"id": 73, "name": "Project 073 - Strategy Focus", "capital": 103, "npv": 123},
    {"id": 74, "name": "Project 074 - Strategy Focus", "capital": 102, "npv": 126},
    {"id": 75, "name": "Project 075 - Strategy Focus", "capital": 141, "npv": 173},
    {"id": 76, "name": "Project 076 - Strategy Focus", "capital": 109, "npv": 152},
    {"id": 77, "name": "Project 077 - Strategy Focus", "capital": 118, "npv": 170},
    {"id": 78, "name": "Project 078 - Strategy Focus", "capital": 93, "npv": 148},
    {"id": 79, "name": "Project 079 - Strategy Focus", "capital": 95, "npv": 142},
    {"id": 80, "name": "Project 080 - Strategy Focus", "capital": 92, "npv": 118},
    {"id": 81, "name": "Project 081 - Strategy Focus", "capital": 71, "npv": 104},
    {"id": 82, "name": "Project 082 - Strategy Focus", "capital": 127, "npv": 159},
    {"id": 83, "name": "Project 083 - Strategy Focus", "capital": 126, "npv": 163},
    {"id": 84, "name": "Project 084 - Strategy Focus", "capital": 123, "npv": 165},
    {"id": 85, "name": "Project 085 - Strategy Focus", "capital": 137, "npv": 164},
    {"id": 86, "name": "Project 086 - Strategy Focus", "capital": 124, "npv": 149},
    {"id": 87, "name": "Project 087 - Strategy Focus", "capital": 103, "npv": 144},
    {"id": 88, "name": "Project 088 - Strategy Focus", "capital": 119, "npv": 166},
    {"id": 89, "name": "Project 089 - Strategy Focus", "capital": 87, "npv": 105},
    {"id": 90, "name": "Project 090 - Strategy Focus", "capital": 87, "npv": 130},
    {"id": 91, "name": "Project 091 - Strategy Focus", "capital": 92, "npv": 138},
    {"id": 92, "name": "Project 092 - Strategy Focus", "capital": 69, "npv": 96},
    {"id": 93, "name": "Project 093 - Strategy Focus", "capital": 149, "npv": 178},
    {"id": 94, "name": "Project 094 - Strategy Focus", "capital": 146, "npv": 193},
    {"id": 95, "name": "Project 095 - Strategy Focus", "capital": 47, "npv": 68},
    {"id": 96, "name": "Project 096 - Strategy Focus", "capital": 12, "npv": 18},
    {"id": 97, "name": "Project 097 - Strategy Focus", "capital": 101, "npv": 131},
    {"id": 98, "name": "Project 098 - Strategy Focus", "capital": 69, "npv": 103},
    {"id": 99, "name": "Project 099 - Strategy Focus", "capital": 46, "npv": 65},
    {"id": 100, "name": "Project 100 - Strategy Focus", "capital": 79, "npv": 108},
    {"id": 101, "name": "Project 101 - Strategy Focus", "capital": 93, "npv": 130},
    {"id": 102, "name": "Project 102 - Strategy Focus", "capital": 49, "npv": 60},
    {"id": 103, "name": "Project 103 - Strategy Focus", "capital": 110, "npv": 157},
    {"id": 104, "name": "Project 104 - Strategy Focus", "capital": 133, "npv": 184},
    {"id": 105, "name": "Project 105 - Strategy Focus", "capital": 92, "npv": 130},
    {"id": 106, "name": "Project 106 - Strategy Focus", "capital": 106, "npv": 128},
    {"id": 107, "name": "Project 107 - Strategy Focus", "capital": 144, "npv": 201},
    {"id": 108, "name": "Project 108 - Strategy Focus", "capital": 124, "npv": 173},
    {"id": 109, "name": "Project 109 - Strategy Focus", "capital": 148, "npv": 189},
    {"id": 110, "name": "Project 110 - Strategy Focus", "capital": 127, "npv": 177}
  ]
}