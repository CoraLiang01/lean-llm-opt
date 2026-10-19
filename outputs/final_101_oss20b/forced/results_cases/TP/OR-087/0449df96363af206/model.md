##### Decision Variables

$x_{pd}\geq0$: quantity of product $p\in P$ produced on day $d\in D$ (units of 100 kg, integer).  
$y_{p}\in\{0,1\}$: binary variable indicating whether product $p$ is activated (produced at least once) during the month.

##### Objective Function

$\displaystyle \max\;\sum_{p\in P}\Bigl(\bigl(\text{price}_{p}-\text{cost}_{p}\bigr)\sum_{d\in D}x_{pd}\Bigr)\;-\;\sum_{p\in P}\text{activation}_{p}\,y_{p}$

##### Constraints

1. **Daily production quota**  
   $\displaystyle \sum_{d\in D}x_{pd}\leq\text{quota}_{p}\cdot|D|,\quad\forall p\in P$  

2. **Demand satisfaction**  
   $\displaystyle \sum_{d\in D}x_{pd}\leq\text{demand}_{p},\quad\forall p\in P$  

3. **Activation linkage**  
   $\displaystyle x_{pd}\leq M\,y_{p},\quad\forall p\in P,\;d\in D$  

4. **Minimum batch size**  
   $\displaystyle x_{pd}\geq\text{batch}_{p}\,y_{p},\quad\forall p\in P,\;d\in D$  

5. **Variable domains**  
   $x_{pd}\in\mathbb{Z}_{+},\;y_{p}\in\{0,1\}$

$P=\{1,2,\dots,80\}$, $D=\{1,2,\dots,22\}$, $M$ is a sufficiently large constant (e.g., $\max_{p}\text{demand}_{p}$).

##### Retrieved Information

{
  "products": [
    {
      "id": 1,
      "demand": 5732,
      "price": 111,
      "cost": 71.38,
      "quota": 586,
      "activation": 178539,
      "batch": 18
    },
    {
      "id": 2,
      "demand": 5607,
      "price": 81,
      "cost": 45.02,
      "quota": 329,
      "activation": 157708,
      "batch": 25
    },
    {
      "id": 3,
      "demand": 4653,
      "price": 145,
      "cost": 107.04,
      "quota": 541,
      "activation": 85192,
      "batch": 23
    },
    {
      "id": 4,
      "demand": 6264,
      "price": 121,
      "cost": 74.03,
      "quota": 589,
      "activation": 178047,
      "batch": 18
    },
    {
      "id": 5,
      "demand": 3835,
      "price": 137,
      "cost": 97.49,
      "quota": 446,
      "activation": 130057,
      "batch": 21
    },
    {
      "id": 6,
      "demand": 3763,
      "price": 115,
      "cost": 77.61,
      "quota": 573,
      "activation": 189259,
      "batch": 28
    },
    {
      "id": 7,
      "demand": 4731,
      "price": 91,
      "cost": 52.27,
      "quota": 521,
      "activation": 153349,
      "batch": 21
    },
    {
      "id": 8,
      "demand": 6431,
      "price": 126,
      "cost": 88.63,
      "quota": 640,
      "activation": 197583,
      "batch": 18
    },
    {
      "id": 9,
      "demand": 4033,
      "price": 80,
      "cost": 47.93,
      "quota": 302,
      "activation": 101440,
      "batch": 17
    },
    {
      "id": 10,
      "demand": 6795,
      "price": 94,
      "cost": 67.71,
      "quota": 369,
      "activation": 178448,
      "batch": 12
    },
    {
      "id": 11,
      "demand": 3277,
      "price": 133,
      "cost": 85.83,
      "quota": 657,
      "activation": 100609,
      "batch": 27
    },
    {
      "id": 12,
      "demand": 4778,
      "price": 92,
      "cost": 66.28,
      "quota": 696,
      "activation": 129100,
      "batch": 21
    },
    {
      "id": 13,
      "demand": 4828,
      "price": 122,
      "cost": 82.12,
      "quota": 344,
      "activation": 87751,
      "batch": 25
    },
    {
      "id": 14,
      "demand": 5647,
      "price": 148,
      "cost": 100.83,
      "quota": 673,
      "activation": 123757,
      "batch": 15
    },
    {
      "id": 15,
      "demand": 6142,
      "price": 86,
      "cost": 53.78,
      "quota": 553,
      "activation": 111228,
      "batch": 17
    },
    {
      "id": 16,
      "demand": 6544,
      "price": 148,
      "cost": 109.38,
      "quota": 622,
      "activation": 116230,
      "batch": 13
    },
    {
      "id": 17,
      "demand": 5648,
      "price": 127,
      "cost": 83.95,
      "quota": 411,
      "activation": 177561,
      "batch": 16
    },
    {
      "id": 18,
      "demand": 6468,
      "price": 83,
      "cost": 50.29,
      "quota": 391,
      "activation": 101758,
      "batch": 14
    },
    {
      "id": 19,
      "demand": 5362,
      "price": 132,
      "cost": 86.01,
      "quota": 641,
      "activation": 125413,
      "batch": 20
    },
    {
      "id": 20,
      "demand": 3705,
      "price": 95,
      "cost": 47.96,
      "quota": 339,
      "activation": 115730,
      "batch": 21
    },
    {
      "id": 21,
      "demand": 6558,
      "price": 100,
      "cost": 57.54,
      "quota": 450,
      "activation": 131668,
      "batch": 29
    },
    {
      "id": 22,
      "demand": 5599,
      "price": 138,
      "cost": 91.78,
      "quota": 445,
      "activation": 144395,
      "batch": 17
    },
    {
      "id": 23,
      "demand": 5135,
      "price": 103,
      "cost": 58.97,
      "quota": 498,
      "activation": 88444,
      "batch": 16
    },
    {
      "id": 24,
      "demand": 5222,
      "price": 93,
      "cost": 60.87,
      "quota": 574,
      "activation": 99946,
      "batch": 20
    },
    {
      "id": 25,
      "demand": 6672,
      "price": 128,
      "cost": 77.72,
      "quota": 648,
      "activation": 183068,
      "batch": 11
    },
    {
      "id": 26,
      "demand": 5897,
      "price": 129,
      "cost": 68.87,
      "quota": 343,
      "activation": 137245,
      "batch": 13
    },
    {
      "id": 27,
      "demand": 4701,
      "price": 149,
      "cost": 85.61,
      "quota": 383,
      "activation": 168462,
      "batch": 15
    },
    {
      "id": 28,
      "demand": 3537,
      "price": 121,
      "cost": 77.74,
      "quota": 597,
      "activation": 195250,
      "batch": 14
    },
    {
      "id": 29,
      "demand": 5893,
      "price": 115,
      "cost": 74.49,
      "quota": 393,
      "activation": 172740,
      "batch": 24
    },
    {
      "id": 30,
      "demand": 5120,
      "price": 144,
      "cost": 92.68,
      "quota": 474,
      "activation": 111959,
      "batch": 16
    },
    {
      "id": 31,
      "demand": 5825,
      "price": 149,
      "cost": 98.83,
      "quota": 501,
      "activation": 170071,
      "batch": 29
    },
    {
      "id": 32,
      "demand": 4940,
      "price": 80,
      "cost": 53.04,
      "quota": 645,
      "activation": 99433,
      "batch": 12
    },
    {
      "id": 33,
      "demand": 5163,
      "price": 130,
      "cost": 79.02,
      "quota": 629,
      "activation": 97137,
      "batch": 26
    },
    {
      "id": 34,
      "demand": 3976,
      "price": 116,
      "cost": 84,
      "quota": 328,
      "activation": 106103,
      "batch": 24
    },
    {
      "id": 35,
      "demand": 3755,
      "price": 114,
      "cost": 67.48,
      "quota": 509,
      "activation": 124510,
      "batch": 22
    },
    {
      "id": 36,
      "demand": 6781,
      "price": 128,
      "cost": 77.95,
      "quota": 405,
      "activation": 131040,
      "batch": 28
    },
    {
      "id": 37,
      "demand": 5046,
      "price": 83,
      "cost": 60.01,
      "quota": 684,
      "activation": 113878,
      "batch": 14
    },
    {
      "id": 38,
      "demand": 4871,
      "price": 122,
      "cost": 85.59,
      "quota": 363,
      "activation": 184589,
      "batch": 25
    },
    {
      "id": 39,
      "demand": 6503,
      "price": 101,
      "cost": 68.27,
      "quota": 316,
      "activation": 179335,
      "batch": 27
    },
    {
      "id": 40,
      "demand": 5496,
      "price": 80,
      "cost": 42,
      "quota": 406,
      "activation": 117183,
      "batch": 17
    },
    {
      "id": 41,
      "demand": 5898,
      "price": 90,
      "cost": 65.69,
      "quota": 464,
      "activation": 197565,
      "batch": 24
    },
    {
      "id": 42,
      "demand": 3099,
      "price": 123,
      "cost": 83.46,
      "quota": 394,
      "activation": 83606,
      "batch": 18
    },
    {
      "id": 43,
      "demand": 5008,
      "price": 138,
      "cost": 103.46,
      "quota": 324,
      "activation": 187833,
      "batch": 28
    },
    {
      "id": 44,
      "demand": 6249,
      "price": 103,
      "cost": 55.35,
      "quota": 416,
      "activation": 148097,
      "batch": 23
    },
    {
      "id": 45,
      "demand": 3755,
      "price": 139,
      "cost": 99.67,
      "quota": 491,
      "activation": 141756,
      "batch": 25
    },
    {
      "id": 46,
      "demand": 3797,
      "price": 82,
      "cost": 44.33,
      "quota": 495,
      "activation": 143697,
      "batch": 19
    },
    {
      "id": 47,
      "demand": 3659,
      "price": 142,
      "cost": 92.85,
      "quota": 607,
      "activation": 175496,
      "batch": 23
    },
    {
      "id": 48,
      "demand": 6219,
      "price": 115,
      "cost": 61.06,
      "quota": 436,
      "activation": 138072,
      "batch": 18
    },
    {
      "id": 49,
      "demand": 5958,
      "price": 147,
      "cost": 104.66,
      "quota": 647,
      "activation": 177933,
      "batch": 27
    },
    {
      "id": 50,
      "demand": 3423,
      "price": 126,
      "cost": 88.43,
      "quota": 393,
      "activation": 119539,
      "batch": 19
    },
    {
      "id": 51,
      "demand": 6360,
      "price": 100,
      "cost": 64.23,
      "quota": 679,
      "activation": 80431,
      "batch": 19
    },
    {
      "id": 52,
      "demand": 6337,
      "price": 130,
      "cost": 78.23,
      "quota": 538,
      "activation": 116842,
      "batch": 24
    },
    {
      "id": 53,
      "demand": 5745,
      "price": 107,
      "cost": 55.35,
      "quota": 387,
      "activation": 193096,
      "batch": 29
    },
    {
      "id": 54,
      "demand": 3639,
      "price": 94,
      "cost": 63.39,
      "quota": 460,
      "activation": 162462,
      "batch": 10
    },
    {
      "id": 55,
      "demand": 3544,
      "price": 121,
      "cost": 74.22,
      "quota": 447,
      "activation": 154823,
      "batch": 21
    },
    {
      "id": 56,
      "demand": 5591,
      "price": 138,
      "cost": 93.91,
      "quota": 372,
      "activation": 192003,
      "batch": 10
    },
    {
      "id": 57,
      "demand": 3714,
      "price": 145,
      "cost": 103.91,
      "quota": 387,
      "activation": 179782,
      "batch": 28
    },
    {
      "id": 58,
      "demand": 5292,
      "price": 116,
      "cost": 86.29,
      "quota": 313,
      "activation": 96277,
      "batch": 18
    },
    {
      "id": 59,
      "demand": 3151,
      "price": 90,
      "cost": 64.26,
      "quota": 614,
      "activation": 140977,
      "batch": 22
    },
    {
      "id": 60,
      "demand": 5723,
      "price": 123,
      "cost": 61.86,
      "quota": 381,
      "activation": 179641,
      "batch": 20
    },
    {
      "id": 61,
      "demand": 5558,
      "price": 91,
      "cost": 53.69,
      "quota": 420,
      "activation": 125571,
      "batch": 18
    },
    {
      "id": 62,
      "demand": 6531,
      "price": 82,
      "cost": 55.96,
      "quota": 672,
      "activation": 136260,
      "batch": 12
    },
    {
      "id": 63,
      "demand": 5930,
      "price": 131,
      "cost": 71.12,
      "quota": 620,
      "activation": 156568,
      "batch": 19
    },
    {
      "id": 64,
      "demand": 4207,
      "price": 112,
      "cost": 70.59,
      "quota": 503,
      "activation": 186993,
      "batch": 15
    },
    {
      "id": 65,
      "demand": 5076,
      "price": 134,
      "cost": 68.82,
      "quota": 520,
      "activation": 114380,
      "batch": 16
    },
    {
      "id": 66,
      "demand": 3802,
      "price": 80,
      "cost": 44,
      "quota": 581,
      "activation": 165018,
      "batch": 16
    },
    {
      "id": 67,
      "demand": 5176,
      "price": 118,
      "cost": 59.55,
      "quota": 588,
      "activation": 81844,
      "batch": 29
    },
    {
      "id": 68,
      "demand": 5176,
      "price": 99,
      "cost": 69.14,
      "quota": 570,
      "activation": 89552,
      "batch": 28
    },
    {
      "id": 69,
      "demand": 4956,
      "price": 126,
      "cost": 70.05,
      "quota": 584,
      "activation": 115437,
      "batch": 11
    },
    {
      "id": 70,
      "demand": 6125,
      "price": 122,
      "cost": 71.53,
      "quota": 576,
      "activation": 175091,
      "batch": 14
    },
    {
      "id": 71,
      "demand": 4925,
      "price": 136,
      "cost": 99.55,
      "quota": 624,
      "activation": 175529,
      "batch": 18
    },
    {
      "id": 72,
      "demand": 6622,
      "price": 140,
      "cost": 94.65,
      "quota": 322,
      "activation": 108114,
      "batch": 27
    },
    {
      "id": 73,
      "demand": 6560,
      "price": 110,
      "cost": 55.88,
      "quota": 527,
      "activation": 80015,
      "batch": 16
    },
    {
      "id": 74,
      "demand": 3756,
      "price": 104,
      "cost": 56.28,
      "quota": 678,
      "activation": 173248,
      "batch": 15
    },
    {
      "id": 75,
      "demand": 3273,
      "price": 82,
      "cost": 53.74,
      "quota": 383,
      "activation": 196932,
      "batch": 11
    },
    {
      "id": 76,
      "demand": 5383,
      "price": 83,
      "cost": 53.48,
      "quota": 435,
      "activation": 92313,
      "batch": 28
    },
    {
      "id": 77,
      "demand": 3388,
      "price": 93,
      "cost": 52.03,
      "quota": 361,
      "activation": 157167,
      "batch": 21
    },
    {
      "id": 78,
      "demand": 4641,
      "price": 120,
      "cost": 88.03,
      "quota": 441,
      "activation": 93783,
      "batch": 15
    },
    {
      "id": 79,
      "demand": 6114,
      "price": 99,
      "cost": 64.7,
      "quota": 305,
      "activation": 155655,
      "batch": 20
    },
    {
      "id": 80,
      "demand": 4466,
      "price": 106,
      "cost": 67.19,
      "quota": 556,
      "activation": 146842,
      "batch": 18
    }
  ],
  "days": 22
}