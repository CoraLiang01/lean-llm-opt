##### Decision Variables

Let $x_i$ be a binary variable for each project $i$ $(i=1,\ldots,110)$:
- $x_i = 1$ if project $i$ is selected, $0$ otherwise.

##### Parameters

Let:
- $c_i$ = Capital investment required for project $i$ (in k$)
- $n_i$ = Expected NPV for project $i$ (in k$)

The data for each project is as follows:

| Project ID | Project Name                        | $c_i$ (k$) | $n_i$ (k$) |
|------------|-------------------------------------|------------|------------|
| 1          | Infrastructure Upgrade              | 50         | 60         |
| 2          | New Product Line A                  | 40         | 50         |
| 3          | Marketing Campaign X                | 30         | 45         |
| 4          | R&D Initiative Alpha                | 25         | 35         |
| 5          | Staff Training Program              | 20         | 28         |
| 6          | System Automation                   | 65         | 75         |
| 7          | Global Expansion Pilot              | 80         | 100        |
| 8          | Green Energy Switch                 | 15         | 20         |
| 9          | Warehouse Optimization              | 48         | 62         |
| 10         | Customer Experience Platform        | 55         | 70         |
| 11         | Project 011 - Strategy Focus        | 87         | 117        |
| 12         | Project 012 - Strategy Focus        | 82         | 109        |
| 13         | Project 013 - Strategy Focus        | 123        | 172        |
| 14         | Project 014 - Strategy Focus        | 137        | 180        |
| 15         | Project 015 - Strategy Focus        | 43         | 62         |
| 16         | Project 016 - Strategy Focus        | 110        | 147        |
| 17         | Project 017 - Strategy Focus        | 94         | 141        |
| 18         | Project 018 - Strategy Focus        | 101        | 151        |
| 19         | Project 019 - Strategy Focus        | 54         | 75         |
| 20         | Project 020 - Strategy Focus        | 143        | 185        |
| 21         | Project 021 - Strategy Focus        | 37         | 51         |
| 22         | Project 022 - Strategy Focus        | 46         | 59         |
| 23         | Project 023 - Strategy Focus        | 81         | 105        |
| 24         | Project 024 - Strategy Focus        | 100        | 123        |
| 25         | Project 025 - Strategy Focus        | 80         | 104        |
| 26         | Project 026 - Strategy Focus        | 102        | 153        |
| 27         | Project 027 - Strategy Focus        | 109        | 163        |
| 28         | Project 028 - Strategy Focus        | 79         | 114        |
| 29         | Project 029 - Strategy Focus        | 100        | 125        |
| 30         | Project 030 - Strategy Focus        | 119        | 160        |
| 31         | Project 031 - Strategy Focus        | 139        | 166        |
| 32         | Project 032 - Strategy Focus        | 11         | 15         |
| 33         | Project 033 - Strategy Focus        | 144        | 184        |
| 34         | Project 034 - Strategy Focus        | 56         | 86         |
| 35         | Project 035 - Strategy Focus        | 59         | 85         |
| 36         | Project 036 - Strategy Focus        | 73         | 109        |
| 37         | Project 037 - Strategy Focus        | 136        | 187        |
| 38         | Project 038 - Strategy Focus        | 85         | 115        |
| 39         | Project 039 - Strategy Focus        | 111        | 155        |
| 40         | Project 040 - Strategy Focus        | 103        | 137        |
| 41         | Project 041 - Strategy Focus        | 146        | 204        |
| 42         | Project 042 - Strategy Focus        | 82         | 98         |
| 43         | Project 043 - Strategy Focus        | 144        | 189        |
| 44         | Project 044 - Strategy Focus        | 143        | 178        |
| 45         | Project 045 - Strategy Focus        | 85         | 130        |
| 46         | Project 046 - Strategy Focus        | 121        | 148        |
| 47         | Project 047 - Strategy Focus        | 147        | 205        |
| 48         | Project 048 - Strategy Focus        | 92         | 128        |
| 49         | Project 049 - Strategy Focus        | 115        | 165        |
| 50         | Project 050 - Strategy Focus        | 13         | 18         |
| 51         | Project 051 - Strategy Focus        | 145        | 203        |
| 52         | Project 052 - Strategy Focus        | 48         | 72         |
| 53         | Project 053 - Strategy Focus        | 74         | 92         |
| 54         | Project 054 - Strategy Focus        | 135        | 167        |
| 55         | Project 055 - Strategy Focus        | 28         | 40         |
| 56         | Project 056 - Strategy Focus        | 74         | 89         |
| 57         | Project 057 - Strategy Focus        | 89         | 111        |
| 58         | Project 058 - Strategy Focus        | 24         | 38         |
| 59         | Project 059 - Strategy Focus        | 141        | 190        |
| 60         | Project 060 - Strategy Focus        | 27         | 33         |
| 61         | Project 061 - Strategy Focus        | 106        | 148        |
| 62         | Project 062 - Strategy Focus        | 37         | 44         |
| 63         | Project 063 - Strategy Focus        | 114        | 167        |
| 64         | Project 064 - Strategy Focus        | 11         | 14         |
| 65         | Project 065 - Strategy Focus        | 113        | 153        |
| 66         | Project 066 - Strategy Focus        | 61         | 91         |
| 67         | Project 067 - Strategy Focus        | 62         | 94         |
| 68         | Project 068 - Strategy Focus        | 145        | 188        |
| 69         | Project 069 - Strategy Focus        | 148        | 177        |
| 70         | Project 070 - Strategy Focus        | 30         | 36         |
| 71         | Project 071 - Strategy Focus        | 134        | 179        |
| 72         | Project 072 - Strategy Focus        | 97         | 143        |
| 73         | Project 073 - Strategy Focus        | 103        | 123        |
| 74         | Project 074 - Strategy Focus        | 102        | 126        |
| 75         | Project 075 - Strategy Focus        | 141        | 173        |
| 76         | Project 076 - Strategy Focus        | 109        | 152        |
| 77         | Project 077 - Strategy Focus        | 118        | 170        |
| 78         | Project 078 - Strategy Focus        | 93         | 148        |
| 79         | Project 079 - Strategy Focus        | 95         | 142        |
| 80         | Project 080 - Strategy Focus        | 92         | 118        |
| 81         | Project 081 - Strategy Focus        | 71         | 104        |
| 82         | Project 082 - Strategy Focus        | 127        | 159        |
| 83         | Project 083 - Strategy Focus        | 126        | 163        |
| 84         | Project 084 - Strategy Focus        | 123        | 165        |
| 85         | Project 085 - Strategy Focus        | 137        | 164        |
| 86         | Project 086 - Strategy Focus        | 124        | 149        |
| 87         | Project 087 - Strategy Focus        | 103        | 144        |
| 88         | Project 088 - Strategy Focus        | 119        | 166        |
| 89         | Project 089 - Strategy Focus        | 87         | 105        |
| 90         | Project 090 - Strategy Focus        | 87         | 130        |
| 91         | Project 091 - Strategy Focus        | 92         | 138        |
| 92         | Project 092 - Strategy Focus        | 69         | 96         |
| 93         | Project 093 - Strategy Focus        | 149        | 178        |
| 94         | Project 094 - Strategy Focus        | 146        | 193        |
| 95         | Project 095 - Strategy Focus        | 47         | 68         |
| 96         | Project 096 - Strategy Focus        | 12         | 18         |
| 97         | Project 097 - Strategy Focus        | 101        | 131        |
| 98         | Project 098 - Strategy Focus        | 69         | 103        |
| 99         | Project 099 - Strategy Focus        | 46         | 65         |
| 100        | Project 100 - Strategy Focus        | 79         | 108        |
| 101        | Project 101 - Strategy Focus        | 93         | 130        |
| 102        | Project 102 - Strategy Focus        | 49         | 60         |
| 103        | Project 103 - Strategy Focus        | 110        | 157        |
| 104        | Project 104 - Strategy Focus        | 133        | 184        |
| 105        | Project 105 - Strategy Focus        | 92         | 130        |
| 106        | Project 106 - Strategy Focus        | 106        | 128        |
| 107        | Project 107 - Strategy Focus        | 144        | 201        |
| 108        | Project 108 - Strategy Focus        | 124        | 173        |
| 109        | Project 109 - Strategy Focus        | 148        | 189        |
| 110        | Project 110 - Strategy Focus        | 127        | 177        |

##### Objective Function

$\max \sum_{i=1}^{110} n_i x_i$

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
   x_i \in \{0,1\} \quad \forall i = 1,\ldots,110
   $$

##### Retrieved Information

{
  "projects": [
    {"Project ID": 1, "Project Name": "Infrastructure Upgrade", "Capital (k$)": 50, "NPV (k$)": 60},
    {"Project ID": 2, "Project Name": "New Product Line A", "Capital (k$)": 40, "NPV (k$)": 50},
    {"Project ID": 3, "Project Name": "Marketing Campaign X", "Capital (k$)": 30, "NPV (k$)": 45},
    {"Project ID": 4, "Project Name": "R&D Initiative Alpha", "Capital (k$)": 25, "NPV (k$)": 35},
    {"Project ID": 5, "Project Name": "Staff Training Program", "Capital (k$)": 20, "NPV (k$)": 28},
    {"Project ID": 6, "Project Name": "System Automation", "Capital (k$)": 65, "NPV (k$)": 75},
    {"Project ID": 7, "Project Name": "Global Expansion Pilot", "Capital (k$)": 80, "NPV (k$)": 100},
    {"Project ID": 8, "Project Name": "Green Energy Switch", "Capital (k$)": 15, "NPV (k$)": 20},
    {"Project ID": 9, "Project Name": "Warehouse Optimization", "Capital (k$)": 48, "NPV (k$)": 62},
    {"Project ID": 10, "Project Name": "Customer Experience Platform", "Capital (k$)": 55, "NPV (k$)": 70},
    {"Project ID": 11, "Project Name": "Project 011 - Strategy Focus", "Capital (k$)": 87, "NPV (k$)": 117},
    {"Project ID": 12, "Project Name": "Project 012 - Strategy Focus", "Capital (k$)": 82, "NPV (k$)": 109},
    {"Project ID": 13, "Project Name": "Project 013 - Strategy Focus", "Capital (k$)": 123, "NPV (k$)": 172},
    {"Project ID": 14, "Project Name": "Project 014 - Strategy Focus", "Capital (k$)": 137, "NPV (k$)": 180},
    {"Project ID": 15, "Project Name": "Project 015 - Strategy Focus", "Capital (k$)": 43, "NPV (k$)": 62},
    {"Project ID": 16, "Project Name": "Project 016 - Strategy Focus", "Capital (k$)": 110, "NPV (k$)": 147},
    {"Project ID": 17, "Project Name": "Project 017 - Strategy Focus", "Capital (k$)": 94, "NPV (k$)": 141},
    {"Project ID": 18, "Project Name": "Project 018 - Strategy Focus", "Capital (k$)": 101, "NPV (k$)": 151},
    {"Project ID": 19, "Project Name": "Project 019 - Strategy Focus", "Capital (k$)": 54, "NPV (k$)": 75},
    {"Project ID": 20, "Project Name": "Project 020 - Strategy Focus", "Capital (k$)": 143, "NPV (k$)": 185},
    {"Project ID": 21, "Project Name": "Project 021 - Strategy Focus", "Capital (k$)": 37, "NPV (k$)": 51},
    {"Project ID": 22, "Project Name": "Project 022 - Strategy Focus", "Capital (k$)": 46, "NPV (k$)": 59},
    {"Project ID": 23, "Project Name": "Project 023 - Strategy Focus", "Capital (k$)": 81, "NPV (k$)": 105},
    {"Project ID": 24, "Project Name": "Project 024 - Strategy Focus", "Capital (k$)": 100, "NPV (k$)": 123},
    {"Project ID": 25, "Project Name": "Project 025 - Strategy Focus", "Capital (k$)": 80, "NPV (k$)": 104},
    {"Project ID": 26, "Project Name": "Project 026 - Strategy Focus", "Capital (k$)": 102, "NPV (k$)": 153},
    {"Project ID": 27, "Project Name": "Project 027 - Strategy Focus", "Capital (k$)": 109, "NPV (k$)": 163},
    {"Project ID": 28, "Project Name": "Project 028 - Strategy Focus", "Capital (k$)": 79, "NPV (k$)": 114},
    {"Project ID": 29, "Project Name": "Project 029 - Strategy Focus", "Capital (k$)": 100, "NPV (k$)": 125},
    {"Project ID": 30, "Project Name": "Project 030 - Strategy Focus", "Capital (k$)": 119, "NPV (k$)": 160},
    {"Project ID": 31, "Project Name": "Project 031 - Strategy Focus", "Capital (k$)": 139, "NPV (k$)": 166},
    {"Project ID": 32, "Project Name": "Project 032 - Strategy Focus", "Capital (k$)": 11, "NPV (k$)": 15},
    {"Project ID": 33, "Project Name": "Project 033 - Strategy Focus", "Capital (k$)": 144, "NPV (k$)": 184},
    {"Project ID": 34, "Project Name": "Project 034 - Strategy Focus", "Capital (k$)": 56, "NPV (k$)": 86},
    {"Project ID": 35, "Project Name": "Project 035 - Strategy Focus", "Capital (k$)": 59, "NPV (k$)": 85},
    {"Project ID": 36, "Project Name": "Project 036 - Strategy Focus", "Capital (k$)": 73, "NPV (k$)": 109},
    {"Project ID": 37, "Project Name": "Project 037 - Strategy Focus", "Capital (k$)": 136, "NPV (k$)": 187},
    {"Project ID": 38, "Project Name": "Project 038 - Strategy Focus", "Capital (k$)": 85, "NPV (k$)": 115},
    {"Project ID": 39, "Project Name": "Project 039 - Strategy Focus", "Capital (k$)": 111, "NPV (k$)": 155},
    {"Project ID": 40, "Project Name": "Project 040 - Strategy Focus", "Capital (k$)": 103, "NPV (k$)": 137},
    {"Project ID": 41, "Project Name": "Project 041 - Strategy Focus", "Capital (k$)": 146, "NPV (k$)": 204},
    {"Project ID": 42, "Project Name": "Project 042 - Strategy Focus", "Capital (k$)": 82, "NPV (k$)": 98},
    {"Project ID": 43, "Project Name": "Project 043 - Strategy Focus", "Capital (k$)": 144, "NPV (k$)": 189},
    {"Project ID": 44, "Project Name": "Project 044 - Strategy Focus", "Capital (k$)": 143, "NPV (k$)": 178},
    {"Project ID": 45, "Project Name": "Project 045 - Strategy Focus", "Capital (k$)": 85, "NPV (k$)": 130},
    {"Project ID": 46, "Project Name": "Project 046 - Strategy Focus", "Capital (k$)": 121, "NPV (k$)": 148},
    {"Project ID": 47, "Project Name": "Project 047 - Strategy Focus", "Capital (k$)": 147, "NPV (k$)": 205},
    {"Project ID": 48, "Project Name": "Project 048 - Strategy Focus", "Capital (k$)": 92, "NPV (k$)": 128},
    {"Project ID": 49, "Project Name": "Project 049 - Strategy Focus", "Capital (k$)": 115, "NPV (k$)": 165},
    {"Project ID": 50, "Project Name": "Project 050 - Strategy Focus", "Capital (k$)": 13, "NPV (k$)": 18},
    {"Project ID": 51, "Project Name": "Project 051 - Strategy Focus", "Capital (k$)": 145, "NPV (k$)": 203},
    {"Project ID": 52, "Project Name": "Project 052 - Strategy Focus", "Capital (k$)": 48, "NPV (k$)": 72},
    {"Project ID": 53, "Project Name": "Project 053 - Strategy Focus", "Capital (k$)": 74, "NPV (k$)": 92},
    {"Project ID": 54, "Project Name": "Project 054 - Strategy Focus", "Capital (k$)": 135, "NPV (k$)": 167},
    {"Project ID": 55, "Project Name": "Project 055 - Strategy Focus", "Capital (k$)": 28, "NPV (k$)": 40},
    {"Project ID": 56, "Project Name": "Project 056 - Strategy Focus", "Capital (k$)": 74, "NPV (k$)": 89},
    {"Project ID": 57, "Project Name": "Project 057 - Strategy Focus", "Capital (k$)": 89, "NPV (k$)": 111},
    {"Project ID": 58, "Project Name": "Project 058 - Strategy Focus", "Capital (k$)": 24, "NPV (k$)": 38},
    {"Project ID": 59, "Project Name": "Project 059 - Strategy Focus", "Capital (k$)": 141, "NPV (k$)": 190},
    {"Project ID": 60, "Project Name": "Project 060 - Strategy Focus", "Capital (k$)": 27, "NPV (k$)": 33},
    {"Project ID": 61, "Project Name": "Project 061 - Strategy Focus", "Capital (k$)": 106, "NPV (k$)": 148},
    {"Project ID": 62, "Project Name": "Project 062 - Strategy Focus", "Capital (k$)": 37, "NPV (k$)": 44},
    {"Project ID": 63, "Project Name": "Project 063 - Strategy Focus", "Capital (k$)": 114, "NPV (k$)": 167},
    {"Project ID": 64, "Project Name": "Project 064 - Strategy Focus", "Capital (k$)": 11, "NPV (k$)": 14},
    {"Project ID": 65, "Project Name": "Project 065 - Strategy Focus", "Capital (k$)": 113, "NPV (k$)": 153},
    {"Project ID": 66, "Project Name": "Project 066 - Strategy Focus", "Capital (k$)": 61, "NPV (k$)": 91},
    {"Project ID": 67, "Project Name": "Project 067 - Strategy Focus", "Capital (k$)": 62, "NPV (k$)": 94},
    {"Project ID": 68, "Project Name": "Project 068 - Strategy Focus", "Capital (k$)": 145, "NPV (k$)": 188},
    {"Project ID": 69, "Project Name": "Project 069 - Strategy Focus", "Capital (k$)": 148, "NPV (k$)": 177},
    {"Project ID": 70, "Project Name": "Project 070 - Strategy Focus", "Capital (k$)": 30, "NPV (k$)": 36},
    {"Project ID": 71, "Project Name": "Project 071 - Strategy Focus", "Capital (k$)": 134, "NPV (k$)": 179},
    {"Project ID": 72, "Project Name": "Project 072 - Strategy Focus", "Capital (k$)": 97, "NPV (k$)": 143},
    {"Project ID": 73, "Project Name": "Project 073 - Strategy Focus", "Capital (k$)": 103, "NPV (k$)": 123},
    {"Project ID": 74, "Project Name": "Project 074 - Strategy Focus", "Capital (k$)": 102, "NPV (k$)": 126},
    {"Project ID": 75, "Project Name": "Project 075 - Strategy Focus", "Capital (k$)": 141, "NPV (k$)": 173},
    {"Project ID": 76, "Project Name": "Project 076 - Strategy Focus", "Capital (k$)": 109, "NPV (k$)": 152},
    {"Project ID": 77, "Project Name": "Project 077 - Strategy Focus", "Capital (k$)": 118, "NPV (k$)": 170},
    {"Project ID": 78, "Project Name": "Project 078 - Strategy Focus", "Capital (k$)": 93, "NPV (k$)": 148},
    {"Project ID": 79, "Project Name": "Project 079 - Strategy Focus", "Capital (k$)": 95, "NPV (k$)": 142},
    {"Project ID": 80, "Project Name": "Project 080 - Strategy Focus", "Capital (k$)": 92, "NPV (k$)": 118},
    {"Project ID": 81, "Project Name": "Project 081 - Strategy Focus", "Capital (k$)": 71, "NPV (k$)": 104},
    {"Project ID": 82, "Project Name": "Project 082 - Strategy Focus", "Capital (k$)": 127, "NPV (k$)": 159},
    {"Project ID": 83, "Project Name": "Project 083 - Strategy Focus", "Capital (k$)": 126, "NPV (k$)": 163},
    {"Project ID": 84, "Project Name": "Project 084 - Strategy Focus", "Capital (k$)": 123, "NPV (k$)": 165},
    {"Project ID": 85, "Project Name": "Project 085 - Strategy Focus", "Capital (k$)": 137, "NPV (k$)": 164},
    {"Project ID": 86, "Project Name": "Project 086 - Strategy Focus", "Capital (k$)": 124, "NPV (k$)": 149},
    {"Project ID": 87, "Project Name": "Project 087 - Strategy Focus", "Capital (k$)": 103, "NPV (k$)": 144},
    {"Project ID": 88, "Project Name": "Project 088 - Strategy Focus", "Capital (k$)": 119, "NPV (k$)": 166},
    {"Project ID": 89, "Project Name": "Project 089 - Strategy Focus", "Capital (k$)": 87, "NPV (k$)": 105},
    {"Project ID": 90, "Project Name": "Project 090 - Strategy Focus", "Capital (k$)": 87, "NPV (k$)": 130},
    {"Project ID": 91, "Project Name": "Project 091 - Strategy Focus", "Capital (k$)": 92, "NPV (k$)": 138},
    {"Project ID": 92, "Project Name": "Project 092 - Strategy Focus", "Capital (k$)": 69, "NPV (k$)": 96},
    {"Project ID": 93, "Project Name": "Project 093 - Strategy Focus", "Capital (k$)": 149, "NPV (k$)": 178},
    {"Project ID": 94, "Project Name": "Project 094 - Strategy Focus", "Capital (k$)": 146, "NPV (k$)": 193},
    {"Project ID": 95, "Project Name": "Project 095 - Strategy Focus", "Capital (k$)": 47, "NPV (k$)": 68},
    {"Project ID": 96, "Project Name": "Project 096 - Strategy Focus", "Capital (k$)": 12, "NPV (k$)": 18},
    {"Project ID": 97, "Project Name": "Project 097 - Strategy Focus", "Capital (k$)": 101, "NPV (k$)": 131},
    {"Project ID": 98, "Project Name": "Project 098 - Strategy Focus", "Capital (k$)": 69, "NPV (k$)": 103},
    {"Project ID": 99, "Project Name": "Project 099 - Strategy Focus", "Capital (k$)": 46, "NPV (k$)": 65},
    {"Project ID": 100, "Project Name": "Project 100 - Strategy Focus", "Capital (k$)": 79, "NPV (k$)": 108},
    {"Project ID": 101, "Project Name": "Project 101 - Strategy Focus", "Capital (k$)": 93, "NPV (k$)": 130},
    {"Project ID": 102, "Project Name": "Project 102 - Strategy Focus", "Capital (k$)": 49, "NPV (k$)": 60},
    {"Project ID": 103, "Project Name": "Project 103 - Strategy Focus", "Capital (k$)": 110, "NPV (k$)": 157},
    {"Project ID": 104, "Project Name": "Project 104 - Strategy Focus", "Capital (k$)": 133, "NPV (k$)": 184},
    {"Project ID": 105, "Project Name": "Project 105 - Strategy Focus", "Capital (k$)": 92, "NPV (k$)": 130},
    {"Project ID": 106, "Project Name": "Project 106 - Strategy Focus", "Capital (k$)": 106, "NPV (k$)": 128},
    {"Project ID": 107, "Project Name": "Project 107 - Strategy Focus", "Capital (k$)": 144, "NPV (k$)": 201},
    {"Project ID": 108, "Project Name": "Project 108 - Strategy Focus", "Capital (k$)": 124, "NPV (k$)": 173},
    {"Project ID": 109, "Project Name": "Project 109 - Strategy Focus", "Capital (k$)": 148, "NPV (k$)": 189},
    {"Project ID": 110, "Project Name": "Project 110 - Strategy Focus", "Capital (k$)": 127, "NPV (k$)": 177}
  ]
}