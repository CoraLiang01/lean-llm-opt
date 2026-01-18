\ Model School_Assignment_Scaled
\ LP format - for model browsing. Use MPS format to capture full model detail.
\ Signature: 0x927f94c02e260591
Minimize
  1.25 x_w_I_N01 + 1.25 x_n_I_N01 + 1.3 x_w_I_N02 + 1.3 x_n_I_N02
   + 1.35 x_w_I_N03 + 1.35 x_n_I_N03 + 1.4 x_w_I_N04 + 1.4 x_n_I_N04
   + 1.45 x_w_I_N05 + 1.45 x_n_I_N05 + 1.5 x_w_I_N06 + 1.5 x_n_I_N06
   + 1.55 x_w_I_N07 + 1.55 x_n_I_N07 + 1.6 x_w_I_N08 + 1.6 x_n_I_N08
   + 1.65 x_w_I_N09 + 1.65 x_n_I_N09 + 1.7 x_w_I_N10 + 1.7 x_n_I_N10
   + 1.75 x_w_I_N11 + 1.75 x_n_I_N11 + 1.8 x_w_I_N12 + 1.8 x_n_I_N12
   + 1.85 x_w_I_N13 + 1.85 x_n_I_N13 + 1.9 x_w_I_N14 + 1.9 x_n_I_N14
   + 1.95 x_w_I_N15 + 1.95 x_n_I_N15 + 2 x_w_I_N16 + 2 x_n_I_N16
   + 3.08 x_w_I_N17 + 3.08 x_n_I_N17 + 3.16 x_w_I_N18 + 3.16 x_n_I_N18
   + 3.24 x_w_I_N19 + 3.24 x_n_I_N19 + 3.32 x_w_I_N20 + 3.32 x_n_I_N20
   + 3.4 x_w_I_N21 + 3.4 x_n_I_N21 + 3.48 x_w_I_N22 + 3.48 x_n_I_N22
   + 3.56 x_w_I_N23 + 3.56 x_n_I_N23 + 3.64 x_w_I_N24 + 3.64 x_n_I_N24
   + 3.72 x_w_I_N25 + 3.72 x_n_I_N25 + 3.8 x_w_I_N26 + 3.8 x_n_I_N26
   + 3.88 x_w_I_N27 + 3.88 x_n_I_N27 + 3.96 x_w_I_N28 + 3.96 x_n_I_N28
   + 4.04 x_w_I_N29 + 4.04 x_n_I_N29 + 4.12 x_w_I_N30 + 4.12 x_n_I_N30
   + 4.2 x_w_I_N31 + 4.2 x_n_I_N31 + 3.08 x_w_II_N01 + 3.08 x_n_II_N01
   + 3.16 x_w_II_N02 + 3.16 x_n_II_N02 + 3.24 x_w_II_N03 + 3.24 x_n_II_N03
   + 3.32 x_w_II_N04 + 3.32 x_n_II_N04 + 3.4 x_w_II_N05 + 3.4 x_n_II_N05
   + 3.48 x_w_II_N06 + 3.48 x_n_II_N06 + 3.56 x_w_II_N07 + 3.56 x_n_II_N07
   + 3.64 x_w_II_N08 + 3.64 x_n_II_N08 + 3.72 x_w_II_N09 + 3.72 x_n_II_N09
   + 3.8 x_w_II_N10 + 3.8 x_n_II_N10 + 3.88 x_w_II_N11 + 3.88 x_n_II_N11
   + 3.96 x_w_II_N12 + 3.96 x_n_II_N12 + 4.04 x_w_II_N13 + 4.04 x_n_II_N13
   + 4.12 x_w_II_N14 + 4.12 x_n_II_N14 + 4.2 x_w_II_N15 + 4.2 x_n_II_N15
   + 4.28 x_w_II_N16 + 4.28 x_n_II_N16 + 1.25 x_w_II_N17 + 1.25 x_n_II_N17
   + 1.3 x_w_II_N18 + 1.3 x_n_II_N18 + 1.35 x_w_II_N19 + 1.35 x_n_II_N19
   + 1.4 x_w_II_N20 + 1.4 x_n_II_N20 + 1.45 x_w_II_N21 + 1.45 x_n_II_N21
   + 1.5 x_w_II_N22 + 1.5 x_n_II_N22 + 1.55 x_w_II_N23 + 1.55 x_n_II_N23
   + 1.6 x_w_II_N24 + 1.6 x_n_II_N24 + 1.65 x_w_II_N25 + 1.65 x_n_II_N25
   + 1.7 x_w_II_N26 + 1.7 x_n_II_N26 + 1.75 x_w_II_N27 + 1.75 x_n_II_N27
   + 1.8 x_w_II_N28 + 1.8 x_n_II_N28 + 1.85 x_w_II_N29 + 1.85 x_n_II_N29
   + 1.9 x_w_II_N30 + 1.9 x_n_II_N30 + 1.95 x_w_II_N31 + 1.95 x_n_II_N31
Subject To
 white_supply[N01]: x_w_I_N01 + x_w_II_N01 = 78
 nonwhite_supply[N01]: x_n_I_N01 + x_n_II_N01 = 22
 white_supply[N02]: x_w_I_N02 + x_w_II_N02 = 57
 nonwhite_supply[N02]: x_n_I_N02 + x_n_II_N02 = 33
 white_supply[N03]: x_w_I_N03 + x_w_II_N03 = 47
 nonwhite_supply[N03]: x_n_I_N03 + x_n_II_N03 = 63
 white_supply[N04]: x_w_I_N04 + x_w_II_N04 = 78
 nonwhite_supply[N04]: x_n_I_N04 + x_n_II_N04 = 22
 white_supply[N05]: x_w_I_N05 + x_w_II_N05 = 57
 nonwhite_supply[N05]: x_n_I_N05 + x_n_II_N05 = 33
 white_supply[N06]: x_w_I_N06 + x_w_II_N06 = 47
 nonwhite_supply[N06]: x_n_I_N06 + x_n_II_N06 = 63
 white_supply[N07]: x_w_I_N07 + x_w_II_N07 = 78
 nonwhite_supply[N07]: x_n_I_N07 + x_n_II_N07 = 22
 white_supply[N08]: x_w_I_N08 + x_w_II_N08 = 57
 nonwhite_supply[N08]: x_n_I_N08 + x_n_II_N08 = 33
 white_supply[N09]: x_w_I_N09 + x_w_II_N09 = 46
 nonwhite_supply[N09]: x_n_I_N09 + x_n_II_N09 = 64
 white_supply[N10]: x_w_I_N10 + x_w_II_N10 = 77
 nonwhite_supply[N10]: x_n_I_N10 + x_n_II_N10 = 23
 white_supply[N11]: x_w_I_N11 + x_w_II_N11 = 56
 nonwhite_supply[N11]: x_n_I_N11 + x_n_II_N11 = 34
 white_supply[N12]: x_w_I_N12 + x_w_II_N12 = 46
 nonwhite_supply[N12]: x_n_I_N12 + x_n_II_N12 = 64
 white_supply[N13]: x_w_I_N13 + x_w_II_N13 = 77
 nonwhite_supply[N13]: x_n_I_N13 + x_n_II_N13 = 23
 white_supply[N14]: x_w_I_N14 + x_w_II_N14 = 56
 nonwhite_supply[N14]: x_n_I_N14 + x_n_II_N14 = 34
 white_supply[N15]: x_w_I_N15 + x_w_II_N15 = 46
 nonwhite_supply[N15]: x_n_I_N15 + x_n_II_N15 = 64
 white_supply[N16]: x_w_I_N16 + x_w_II_N16 = 77
 nonwhite_supply[N16]: x_n_I_N16 + x_n_II_N16 = 23
 white_supply[N17]: x_w_I_N17 + x_w_II_N17 = 56
 nonwhite_supply[N17]: x_n_I_N17 + x_n_II_N17 = 34
 white_supply[N18]: x_w_I_N18 + x_w_II_N18 = 46
 nonwhite_supply[N18]: x_n_I_N18 + x_n_II_N18 = 64
 white_supply[N19]: x_w_I_N19 + x_w_II_N19 = 77
 nonwhite_supply[N19]: x_n_I_N19 + x_n_II_N19 = 23
 white_supply[N20]: x_w_I_N20 + x_w_II_N20 = 56
 nonwhite_supply[N20]: x_n_I_N20 + x_n_II_N20 = 34
 white_supply[N21]: x_w_I_N21 + x_w_II_N21 = 46
 nonwhite_supply[N21]: x_n_I_N21 + x_n_II_N21 = 64
 white_supply[N22]: x_w_I_N22 + x_w_II_N22 = 77
 nonwhite_supply[N22]: x_n_I_N22 + x_n_II_N22 = 23
 white_supply[N23]: x_w_I_N23 + x_w_II_N23 = 56
 nonwhite_supply[N23]: x_n_I_N23 + x_n_II_N23 = 34
 white_supply[N24]: x_w_I_N24 + x_w_II_N24 = 46
 nonwhite_supply[N24]: x_n_I_N24 + x_n_II_N24 = 64
 white_supply[N25]: x_w_I_N25 + x_w_II_N25 = 77
 nonwhite_supply[N25]: x_n_I_N25 + x_n_II_N25 = 23
 white_supply[N26]: x_w_I_N26 + x_w_II_N26 = 56
 nonwhite_supply[N26]: x_n_I_N26 + x_n_II_N26 = 34
 white_supply[N27]: x_w_I_N27 + x_w_II_N27 = 46
 nonwhite_supply[N27]: x_n_I_N27 + x_n_II_N27 = 64
 white_supply[N28]: x_w_I_N28 + x_w_II_N28 = 77
 nonwhite_supply[N28]: x_n_I_N28 + x_n_II_N28 = 23
 white_supply[N29]: x_w_I_N29 + x_w_II_N29 = 56
 nonwhite_supply[N29]: x_n_I_N29 + x_n_II_N29 = 34
 white_supply[N30]: x_w_I_N30 + x_w_II_N30 = 46
 nonwhite_supply[N30]: x_n_I_N30 + x_n_II_N30 = 64
 white_supply[N31]: x_w_I_N31 + x_w_II_N31 = 74
 nonwhite_supply[N31]: x_n_I_N31 + x_n_II_N31 = 46
 capacity[I]: x_w_I_N01 + x_n_I_N01 + x_w_I_N02 + x_n_I_N02 + x_w_I_N03
   + x_n_I_N03 + x_w_I_N04 + x_n_I_N04 + x_w_I_N05 + x_n_I_N05 + x_w_I_N06
   + x_n_I_N06 + x_w_I_N07 + x_n_I_N07 + x_w_I_N08 + x_n_I_N08 + x_w_I_N09
   + x_n_I_N09 + x_w_I_N10 + x_n_I_N10 + x_w_I_N11 + x_n_I_N11 + x_w_I_N12
   + x_n_I_N12 + x_w_I_N13 + x_n_I_N13 + x_w_I_N14 + x_n_I_N14 + x_w_I_N15
   + x_n_I_N15 + x_w_I_N16 + x_n_I_N16 + x_w_I_N17 + x_n_I_N17 + x_w_I_N18
   + x_n_I_N18 + x_w_I_N19 + x_n_I_N19 + x_w_I_N20 + x_n_I_N20 + x_w_I_N21
   + x_n_I_N21 + x_w_I_N22 + x_n_I_N22 + x_w_I_N23 + x_n_I_N23 + x_w_I_N24
   + x_n_I_N24 + x_w_I_N25 + x_n_I_N25 + x_w_I_N26 + x_n_I_N26 + x_w_I_N27
   + x_n_I_N27 + x_w_I_N28 + x_n_I_N28 + x_w_I_N29 + x_n_I_N29 + x_w_I_N30
   + x_n_I_N30 + x_w_I_N31 + x_n_I_N31 <= 2028
 capacity[II]: x_w_II_N01 + x_n_II_N01 + x_w_II_N02 + x_n_II_N02
   + x_w_II_N03 + x_n_II_N03 + x_w_II_N04 + x_n_II_N04 + x_w_II_N05
   + x_n_II_N05 + x_w_II_N06 + x_n_II_N06 + x_w_II_N07 + x_n_II_N07
   + x_w_II_N08 + x_n_II_N08 + x_w_II_N09 + x_n_II_N09 + x_w_II_N10
   + x_n_II_N10 + x_w_II_N11 + x_n_II_N11 + x_w_II_N12 + x_n_II_N12
   + x_w_II_N13 + x_n_II_N13 + x_w_II_N14 + x_n_II_N14 + x_w_II_N15
   + x_n_II_N15 + x_w_II_N16 + x_n_II_N16 + x_w_II_N17 + x_n_II_N17
   + x_w_II_N18 + x_n_II_N18 + x_w_II_N19 + x_n_II_N19 + x_w_II_N20
   + x_n_II_N20 + x_w_II_N21 + x_n_II_N21 + x_w_II_N22 + x_n_II_N22
   + x_w_II_N23 + x_n_II_N23 + x_w_II_N24 + x_n_II_N24 + x_w_II_N25
   + x_n_II_N25 + x_w_II_N26 + x_n_II_N26 + x_w_II_N27 + x_n_II_N27
   + x_w_II_N28 + x_n_II_N28 + x_w_II_N29 + x_n_II_N29 + x_w_II_N30
   + x_n_II_N30 + x_w_II_N31 + x_n_II_N31 <= 1560
 white_min[I]: 0.5 x_w_I_N01 - 0.5 x_n_I_N01 + 0.5 x_w_I_N02
   - 0.5 x_n_I_N02 + 0.5 x_w_I_N03 - 0.5 x_n_I_N03 + 0.5 x_w_I_N04
   - 0.5 x_n_I_N04 + 0.5 x_w_I_N05 - 0.5 x_n_I_N05 + 0.5 x_w_I_N06
   - 0.5 x_n_I_N06 + 0.5 x_w_I_N07 - 0.5 x_n_I_N07 + 0.5 x_w_I_N08
   - 0.5 x_n_I_N08 + 0.5 x_w_I_N09 - 0.5 x_n_I_N09 + 0.5 x_w_I_N10
   - 0.5 x_n_I_N10 + 0.5 x_w_I_N11 - 0.5 x_n_I_N11 + 0.5 x_w_I_N12
   - 0.5 x_n_I_N12 + 0.5 x_w_I_N13 - 0.5 x_n_I_N13 + 0.5 x_w_I_N14
   - 0.5 x_n_I_N14 + 0.5 x_w_I_N15 - 0.5 x_n_I_N15 + 0.5 x_w_I_N16
   - 0.5 x_n_I_N16 + 0.5 x_w_I_N17 - 0.5 x_n_I_N17 + 0.5 x_w_I_N18
   - 0.5 x_n_I_N18 + 0.5 x_w_I_N19 - 0.5 x_n_I_N19 + 0.5 x_w_I_N20
   - 0.5 x_n_I_N20 + 0.5 x_w_I_N21 - 0.5 x_n_I_N21 + 0.5 x_w_I_N22
   - 0.5 x_n_I_N22 + 0.5 x_w_I_N23 - 0.5 x_n_I_N23 + 0.5 x_w_I_N24
   - 0.5 x_n_I_N24 + 0.5 x_w_I_N25 - 0.5 x_n_I_N25 + 0.5 x_w_I_N26
   - 0.5 x_n_I_N26 + 0.5 x_w_I_N27 - 0.5 x_n_I_N27 + 0.5 x_w_I_N28
   - 0.5 x_n_I_N28 + 0.5 x_w_I_N29 - 0.5 x_n_I_N29 + 0.5 x_w_I_N30
   - 0.5 x_n_I_N30 + 0.5 x_w_I_N31 - 0.5 x_n_I_N31 >= 0
 white_max[I]: 0.3 x_w_I_N01 - 0.7 x_n_I_N01 + 0.3 x_w_I_N02
   - 0.7 x_n_I_N02 + 0.3 x_w_I_N03 - 0.7 x_n_I_N03 + 0.3 x_w_I_N04
   - 0.7 x_n_I_N04 + 0.3 x_w_I_N05 - 0.7 x_n_I_N05 + 0.3 x_w_I_N06
   - 0.7 x_n_I_N06 + 0.3 x_w_I_N07 - 0.7 x_n_I_N07 + 0.3 x_w_I_N08
   - 0.7 x_n_I_N08 + 0.3 x_w_I_N09 - 0.7 x_n_I_N09 + 0.3 x_w_I_N10
   - 0.7 x_n_I_N10 + 0.3 x_w_I_N11 - 0.7 x_n_I_N11 + 0.3 x_w_I_N12
   - 0.7 x_n_I_N12 + 0.3 x_w_I_N13 - 0.7 x_n_I_N13 + 0.3 x_w_I_N14
   - 0.7 x_n_I_N14 + 0.3 x_w_I_N15 - 0.7 x_n_I_N15 + 0.3 x_w_I_N16
   - 0.7 x_n_I_N16 + 0.3 x_w_I_N17 - 0.7 x_n_I_N17 + 0.3 x_w_I_N18
   - 0.7 x_n_I_N18 + 0.3 x_w_I_N19 - 0.7 x_n_I_N19 + 0.3 x_w_I_N20
   - 0.7 x_n_I_N20 + 0.3 x_w_I_N21 - 0.7 x_n_I_N21 + 0.3 x_w_I_N22
   - 0.7 x_n_I_N22 + 0.3 x_w_I_N23 - 0.7 x_n_I_N23 + 0.3 x_w_I_N24
   - 0.7 x_n_I_N24 + 0.3 x_w_I_N25 - 0.7 x_n_I_N25 + 0.3 x_w_I_N26
   - 0.7 x_n_I_N26 + 0.3 x_w_I_N27 - 0.7 x_n_I_N27 + 0.3 x_w_I_N28
   - 0.7 x_n_I_N28 + 0.3 x_w_I_N29 - 0.7 x_n_I_N29 + 0.3 x_w_I_N30
   - 0.7 x_n_I_N30 + 0.3 x_w_I_N31 - 0.7 x_n_I_N31 <= 0
 white_min[II]: 0.5 x_w_II_N01 - 0.5 x_n_II_N01 + 0.5 x_w_II_N02
   - 0.5 x_n_II_N02 + 0.5 x_w_II_N03 - 0.5 x_n_II_N03 + 0.5 x_w_II_N04
   - 0.5 x_n_II_N04 + 0.5 x_w_II_N05 - 0.5 x_n_II_N05 + 0.5 x_w_II_N06
   - 0.5 x_n_II_N06 + 0.5 x_w_II_N07 - 0.5 x_n_II_N07 + 0.5 x_w_II_N08
   - 0.5 x_n_II_N08 + 0.5 x_w_II_N09 - 0.5 x_n_II_N09 + 0.5 x_w_II_N10
   - 0.5 x_n_II_N10 + 0.5 x_w_II_N11 - 0.5 x_n_II_N11 + 0.5 x_w_II_N12
   - 0.5 x_n_II_N12 + 0.5 x_w_II_N13 - 0.5 x_n_II_N13 + 0.5 x_w_II_N14
   - 0.5 x_n_II_N14 + 0.5 x_w_II_N15 - 0.5 x_n_II_N15 + 0.5 x_w_II_N16
   - 0.5 x_n_II_N16 + 0.5 x_w_II_N17 - 0.5 x_n_II_N17 + 0.5 x_w_II_N18
   - 0.5 x_n_II_N18 + 0.5 x_w_II_N19 - 0.5 x_n_II_N19 + 0.5 x_w_II_N20
   - 0.5 x_n_II_N20 + 0.5 x_w_II_N21 - 0.5 x_n_II_N21 + 0.5 x_w_II_N22
   - 0.5 x_n_II_N22 + 0.5 x_w_II_N23 - 0.5 x_n_II_N23 + 0.5 x_w_II_N24
   - 0.5 x_n_II_N24 + 0.5 x_w_II_N25 - 0.5 x_n_II_N25 + 0.5 x_w_II_N26
   - 0.5 x_n_II_N26 + 0.5 x_w_II_N27 - 0.5 x_n_II_N27 + 0.5 x_w_II_N28
   - 0.5 x_n_II_N28 + 0.5 x_w_II_N29 - 0.5 x_n_II_N29 + 0.5 x_w_II_N30
   - 0.5 x_n_II_N30 + 0.5 x_w_II_N31 - 0.5 x_n_II_N31 >= 0
 white_max[II]: 0.3 x_w_II_N01 - 0.7 x_n_II_N01 + 0.3 x_w_II_N02
   - 0.7 x_n_II_N02 + 0.3 x_w_II_N03 - 0.7 x_n_II_N03 + 0.3 x_w_II_N04
   - 0.7 x_n_II_N04 + 0.3 x_w_II_N05 - 0.7 x_n_II_N05 + 0.3 x_w_II_N06
   - 0.7 x_n_II_N06 + 0.3 x_w_II_N07 - 0.7 x_n_II_N07 + 0.3 x_w_II_N08
   - 0.7 x_n_II_N08 + 0.3 x_w_II_N09 - 0.7 x_n_II_N09 + 0.3 x_w_II_N10
   - 0.7 x_n_II_N10 + 0.3 x_w_II_N11 - 0.7 x_n_II_N11 + 0.3 x_w_II_N12
   - 0.7 x_n_II_N12 + 0.3 x_w_II_N13 - 0.7 x_n_II_N13 + 0.3 x_w_II_N14
   - 0.7 x_n_II_N14 + 0.3 x_w_II_N15 - 0.7 x_n_II_N15 + 0.3 x_w_II_N16
   - 0.7 x_n_II_N16 + 0.3 x_w_II_N17 - 0.7 x_n_II_N17 + 0.3 x_w_II_N18
   - 0.7 x_n_II_N18 + 0.3 x_w_II_N19 - 0.7 x_n_II_N19 + 0.3 x_w_II_N20
   - 0.7 x_n_II_N20 + 0.3 x_w_II_N21 - 0.7 x_n_II_N21 + 0.3 x_w_II_N22
   - 0.7 x_n_II_N22 + 0.3 x_w_II_N23 - 0.7 x_n_II_N23 + 0.3 x_w_II_N24
   - 0.7 x_n_II_N24 + 0.3 x_w_II_N25 - 0.7 x_n_II_N25 + 0.3 x_w_II_N26
   - 0.7 x_n_II_N26 + 0.3 x_w_II_N27 - 0.7 x_n_II_N27 + 0.3 x_w_II_N28
   - 0.7 x_n_II_N28 + 0.3 x_w_II_N29 - 0.7 x_n_II_N29 + 0.3 x_w_II_N30
   - 0.7 x_n_II_N30 + 0.3 x_w_II_N31 - 0.7 x_n_II_N31 <= 0
Bounds
Generals
 x_w_I_N01 x_n_I_N01 x_w_I_N02 x_n_I_N02 x_w_I_N03 x_n_I_N03 x_w_I_N04
 x_n_I_N04 x_w_I_N05 x_n_I_N05 x_w_I_N06 x_n_I_N06 x_w_I_N07 x_n_I_N07
 x_w_I_N08 x_n_I_N08 x_w_I_N09 x_n_I_N09 x_w_I_N10 x_n_I_N10 x_w_I_N11
 x_n_I_N11 x_w_I_N12 x_n_I_N12 x_w_I_N13 x_n_I_N13 x_w_I_N14 x_n_I_N14
 x_w_I_N15 x_n_I_N15 x_w_I_N16 x_n_I_N16 x_w_I_N17 x_n_I_N17 x_w_I_N18
 x_n_I_N18 x_w_I_N19 x_n_I_N19 x_w_I_N20 x_n_I_N20 x_w_I_N21 x_n_I_N21
 x_w_I_N22 x_n_I_N22 x_w_I_N23 x_n_I_N23 x_w_I_N24 x_n_I_N24 x_w_I_N25
 x_n_I_N25 x_w_I_N26 x_n_I_N26 x_w_I_N27 x_n_I_N27 x_w_I_N28 x_n_I_N28
 x_w_I_N29 x_n_I_N29 x_w_I_N30 x_n_I_N30 x_w_I_N31 x_n_I_N31 x_w_II_N01
 x_n_II_N01 x_w_II_N02 x_n_II_N02 x_w_II_N03 x_n_II_N03 x_w_II_N04
 x_n_II_N04 x_w_II_N05 x_n_II_N05 x_w_II_N06 x_n_II_N06 x_w_II_N07
 x_n_II_N07 x_w_II_N08 x_n_II_N08 x_w_II_N09 x_n_II_N09 x_w_II_N10
 x_n_II_N10 x_w_II_N11 x_n_II_N11 x_w_II_N12 x_n_II_N12 x_w_II_N13
 x_n_II_N13 x_w_II_N14 x_n_II_N14 x_w_II_N15 x_n_II_N15 x_w_II_N16
 x_n_II_N16 x_w_II_N17 x_n_II_N17 x_w_II_N18 x_n_II_N18 x_w_II_N19
 x_n_II_N19 x_w_II_N20 x_n_II_N20 x_w_II_N21 x_n_II_N21 x_w_II_N22
 x_n_II_N22 x_w_II_N23 x_n_II_N23 x_w_II_N24 x_n_II_N24 x_w_II_N25
 x_n_II_N25 x_w_II_N26 x_n_II_N26 x_w_II_N27 x_n_II_N27 x_w_II_N28
 x_n_II_N28 x_w_II_N29 x_n_II_N29 x_w_II_N30 x_n_II_N30 x_w_II_N31
 x_n_II_N31
End
