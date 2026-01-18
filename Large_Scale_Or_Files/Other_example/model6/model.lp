\ Model WageDetermination_Large
\ LP format - for model browsing. Use MPS format to capture full model detail.
\ Signature: 0x427d6044d94e6bfe
Minimize
 
Subject To
 Fix_Carpenter: wage[0] = 60
 Balance[0]: - 9 wage[0] + wage[12] + wage[45] + wage[50] + wage[65]
   + wage[68] + wage[76] + wage[85] + wage[88] + wage[100] + wage[104]
   + wage[118] + wage[119] + wage[124] + wage[125] + wage[149] = 0
 Balance[1]: wage[0] - 9 wage[1] + wage[2] + wage[6] + wage[20] + wage[26]
   + wage[50] + wage[132] = 0
 Balance[2]: wage[1] - 9 wage[2] + wage[17] + wage[23] + wage[68]
   + wage[75] + wage[77] + wage[90] + wage[101] = 0
 Balance[3]: wage[2] - 9 wage[3] + wage[7] + wage[32] + wage[50] + wage[76]
   + wage[95] + wage[97] + wage[135] = 0
 Balance[4]: wage[3] - 9 wage[4] + wage[11] + wage[13] + wage[41]
   + wage[62] + wage[72] + wage[84] + wage[101] + wage[106] + wage[111]
   + wage[112] + wage[133] = 0
 Balance[5]: wage[4] - 9 wage[5] + wage[33] + wage[42] + wage[74]
   + wage[82] + wage[94] + wage[97] = 0
 Balance[6]: wage[5] - 9 wage[6] + wage[25] + wage[61] + wage[90]
   + wage[130] + wage[131] + wage[149] = 0
 Balance[7]: 2 wage[6] - 9 wage[7] + wage[31] + wage[38] + wage[41]
   + wage[53] + wage[69] + wage[83] + wage[88] + wage[93] + wage[126]
   + wage[128] + wage[129] + wage[146] = 0
 Balance[8]: wage[5] + wage[7] - 9 wage[8] + wage[9] + wage[26] + wage[58]
   + wage[61] + wage[103] + wage[135] = 0
 Balance[9]: wage[8] - 9 wage[9] + wage[49] + wage[71] + wage[101]
   + wage[111] + wage[127] + wage[148] = 0
 Balance[10]: wage[9] - 9 wage[10] + wage[24] + wage[72] + wage[105]
   + wage[114] + wage[118] + wage[132] + 2 wage[141] = 0
 Balance[11]: wage[10] - 8 wage[11] + wage[19] + wage[51] + wage[52]
   + wage[55] + wage[68] + wage[77] + wage[85] + wage[139] + wage[145]
   + wage[146] + wage[147] = 0
 Balance[12]: wage[11] - 9 wage[12] + wage[15] + wage[37] + wage[43]
   + wage[55] + wage[92] + wage[102] = 0
 Balance[13]: wage[5] + wage[7] + wage[12] - 9 wage[13] + wage[67]
   + 2 wage[89] + wage[106] + wage[108] + wage[135] + wage[149] = 0
 Balance[14]: wage[0] + wage[3] + wage[8] + wage[9] + 2 wage[13]
   - 9 wage[14] + wage[40] + wage[91] + wage[110] + wage[112] = 0
 Balance[15]: wage[14] - 9 wage[15] + wage[46] + wage[50] + wage[69]
   + wage[148] = 0
 Balance[16]: wage[15] - 9 wage[16] + wage[33] + wage[36] + wage[56]
   + wage[81] + wage[91] + wage[102] + wage[121] + wage[137] + wage[143]
   + wage[145] = 0
 Balance[17]: wage[4] + wage[16] - 9 wage[17] + wage[35] + wage[90]
   + wage[125] = 0
 Balance[18]: wage[17] - 9 wage[18] + wage[44] + wage[57] + wage[63]
   + wage[64] + wage[99] + wage[125] + wage[145] = 0
 Balance[19]: wage[18] - 9 wage[19] + wage[38] + wage[67] + wage[99]
   + wage[133] = 0
 Balance[20]: wage[0] + wage[2] + wage[4] + wage[19] - 9 wage[20]
   + wage[29] + wage[31] + wage[54] + wage[84] + wage[95] = 0
 Balance[21]: wage[2] + wage[20] - 9 wage[21] + wage[31] + wage[47]
   + wage[85] + wage[135] = 0
 Balance[22]: wage[21] - 9 wage[22] + wage[44] + wage[49] + wage[56]
   + wage[75] + wage[108] + wage[115] = 0
 Balance[23]: wage[8] + 2 wage[22] - 9 wage[23] + wage[34] + wage[35]
   + wage[50] + wage[134] = 0
 Balance[24]: wage[23] - 9 wage[24] + wage[35] + wage[59] + wage[83]
   + 2 wage[92] + wage[120] + wage[122] + wage[135] + wage[144] + wage[147]
   = 0
 Balance[25]: wage[24] - 9 wage[25] + wage[37] + wage[73] + wage[86]
   + wage[88] + wage[94] + wage[98] + wage[113] + wage[140] + 2 wage[148]
   = 0
 Balance[26]: wage[13] + wage[17] + wage[25] - 9 wage[26] + 2 wage[31]
   + wage[61] + wage[73] + wage[114] + wage[138] = 0
 Balance[27]: wage[11] + wage[15] + 2 wage[16] + wage[26] - 9 wage[27]
   + wage[32] + wage[58] + wage[70] + wage[89] + wage[91] = 0
 Balance[28]: wage[14] + wage[27] - 9 wage[28] + wage[47] + wage[52]
   + wage[62] + wage[81] + wage[118] + wage[122] + wage[124] + wage[132]
   + wage[136] + wage[140] = 0
 Balance[29]: wage[16] + wage[28] - 9 wage[29] + wage[32] + wage[56]
   + wage[71] + wage[79] + wage[80] + wage[106] + wage[125] = 0
 Balance[30]: wage[29] - 9 wage[30] + wage[96] + wage[117] + wage[138] = 0
 Balance[31]: wage[30] - 9 wage[31] + wage[36] + wage[39] + wage[76]
   + wage[108] + wage[124] + 2 wage[127] + wage[130] + wage[148] = 0
 Balance[32]: wage[10] + wage[11] + wage[29] + wage[30] + wage[31]
   - 9 wage[32] + wage[48] + wage[51] + wage[72] + wage[95] + 2 wage[110]
   + wage[142] = 0
 Balance[33]: wage[32] - 9 wage[33] + wage[45] + wage[49] + wage[76]
   + wage[82] + wage[86] + wage[93] + wage[137] = 0
 Balance[34]: wage[6] + wage[12] + 2 wage[33] - 9 wage[34] + wage[53]
   + wage[71] + wage[104] + wage[106] + wage[120] + wage[129] + wage[144]
   = 0
 Balance[35]: wage[15] + wage[34] - 9 wage[35] + wage[38] + wage[55]
   + wage[63] + wage[88] + wage[97] + wage[113] + wage[131] + wage[140]
   + wage[142] = 0
 Balance[36]: wage[12] + wage[26] + wage[34] + wage[35] - 9 wage[36]
   + wage[55] + wage[100] + wage[116] + wage[133] + wage[136] = 0
 Balance[37]: wage[2] + wage[21] + wage[31] + wage[36] - 9 wage[37]
   + wage[79] + wage[90] + wage[100] + wage[103] + wage[110] + wage[112]
   + wage[114] + wage[121] = 0
 Balance[38]: wage[19] + wage[24] + 2 wage[37] - 9 wage[38] + wage[48]
   + wage[60] + wage[64] + wage[77] + wage[91] + wage[108] + wage[109]
   + wage[141] + wage[144] = 0
 Balance[39]: wage[8] + wage[38] - 9 wage[39] + wage[96] + wage[104]
   + wage[107] + wage[118] + wage[135] + wage[141] + wage[143] = 0
 Balance[40]: wage[9] + wage[11] + wage[39] - 9 wage[40] + wage[70]
   + wage[118] + wage[137] = 0
 Balance[41]: wage[13] + wage[25] + wage[40] - 9 wage[41] + wage[60]
   + wage[94] + wage[107] + wage[110] + wage[137] + wage[144] = 0
 Balance[42]: wage[37] + wage[41] - 9 wage[42] + wage[52] + wage[82]
   + wage[113] = 0
 Balance[43]: wage[7] + wage[16] + wage[42] - 9 wage[43] + wage[52]
   + wage[66] + wage[91] + wage[98] + wage[145] = 0
 Balance[44]: wage[9] + wage[15] + wage[43] - 9 wage[44] + wage[63]
   + wage[86] = 0
 Balance[45]: wage[33] + wage[34] + wage[44] - 9 wage[45] + wage[52]
   + wage[70] + wage[86] + wage[115] + wage[118] + wage[130] + wage[139]
   = 0
 Balance[46]: wage[42] + wage[45] - 9 wage[46] + wage[48] + wage[66]
   + wage[67] + wage[71] + wage[78] + wage[86] + wage[88] + wage[99]
   + 2 wage[112] + wage[115] + wage[132] + wage[146] + wage[148] = 0
 Balance[47]: wage[8] + wage[11] + wage[26] + wage[27] + wage[29]
   + wage[32] + 2 wage[46] - 9 wage[47] + wage[87] = 0
 Balance[48]: wage[3] + wage[33] + wage[40] + wage[47] - 9 wage[48]
   + wage[67] + wage[103] + wage[122] + wage[124] + wage[125] + wage[145]
   = 0
 Balance[49]: wage[6] + wage[37] + wage[48] - 9 wage[49] + wage[77]
   + wage[79] = 0
 Balance[50]: wage[3] + wage[4] + wage[18] + wage[31] + wage[39] + wage[49]
   - 9 wage[50] + wage[76] + wage[80] + wage[84] + wage[117] + wage[129]
   + wage[140] = 0
 Balance[51]: wage[14] + wage[19] + wage[24] + wage[42] + wage[50]
   - 9 wage[51] + wage[58] + wage[80] = 0
 Balance[52]: wage[1] + wage[5] + wage[8] + wage[21] + wage[41] + wage[51]
   - 9 wage[52] + wage[59] + wage[75] + wage[82] + wage[109] + wage[120]
   + wage[133] + wage[139] = 0
 Balance[53]: 2 wage[7] + wage[20] + wage[27] + wage[29] + wage[35]
   + wage[41] + wage[52] - 9 wage[53] + wage[68] + wage[77] + wage[128]
   + wage[140] + wage[141] = 0
 Balance[54]: wage[3] + wage[42] + wage[53] - 9 wage[54] + wage[74]
   + wage[96] + wage[119] + wage[126] + wage[127] + wage[130] + wage[142]
   + wage[146] = 0
 Balance[55]: wage[54] - 9 wage[55] + wage[66] + wage[71] + wage[87]
   + 2 wage[89] + wage[142] = 0
 Balance[56]: wage[38] + wage[55] - 9 wage[56] + wage[59] + wage[78]
   + wage[119] + wage[120] + wage[128] + wage[138] + wage[145] = 0
 Balance[57]: wage[2] + wage[35] + 2 wage[40] + wage[44] + wage[45]
   + wage[54] + wage[56] - 9 wage[57] + wage[64] + wage[66] = 0
 Balance[58]: wage[3] + wage[18] + wage[57] - 9 wage[58] + wage[73]
   + wage[94] + wage[134] + wage[147] = 0
 Balance[59]: wage[4] + wage[25] + wage[37] + wage[41] + wage[58]
   - 9 wage[59] + wage[70] + wage[105] + wage[106] + wage[116] = 0
 Balance[60]: wage[32] + wage[59] - 9 wage[60] + wage[65] + wage[66]
   + wage[84] + wage[85] + wage[87] + wage[143] = 0
 Balance[61]: wage[12] + wage[15] + wage[18] + wage[19] + wage[44]
   + wage[56] + wage[60] - 9 wage[61] + wage[96] + wage[98] + wage[103]
   + wage[107] + wage[124] + wage[146] = 0
 Balance[62]: wage[10] + wage[14] + wage[39] + wage[44] + wage[61]
   - 9 wage[62] + wage[73] + wage[87] + wage[102] + wage[108] + wage[119]
   + wage[120] + wage[122] + wage[123] + wage[126] = 0
 Balance[63]: wage[3] + wage[62] - 9 wage[63] + wage[94] + wage[100]
   + 2 wage[105] + wage[139] = 0
 Balance[64]: wage[9] + wage[38] + wage[51] + wage[62] + wage[63]
   - 9 wage[64] + wage[72] + wage[102] + wage[128] = 0
 Balance[65]: wage[15] + wage[39] + wage[64] - 9 wage[65] + wage[78]
   + wage[97] + wage[111] = 0
 Balance[66]: wage[34] + wage[35] + wage[51] + wage[65] - 9 wage[66]
   + wage[92] + wage[109] + wage[128] + wage[129] = 0
 Balance[67]: wage[21] + wage[29] + wage[66] - 9 wage[67] + wage[92]
   + wage[99] + wage[113] + wage[116] = 0
 Balance[68]: wage[22] + 2 wage[47] + wage[49] + wage[50] + wage[59]
   + wage[67] - 9 wage[68] + wage[81] + wage[103] + wage[114] + wage[142]
   = 0
 Balance[69]: wage[23] + wage[40] + wage[63] + wage[68] - 9 wage[69]
   + wage[85] + wage[95] + wage[99] + wage[136] = 0
 Balance[70]: wage[9] + wage[15] + wage[63] + wage[69] - 9 wage[70]
   + wage[124] = 0
 Balance[71]: wage[0] + wage[11] + wage[70] - 9 wage[71] + wage[77]
   + wage[105] + wage[107] + wage[117] + wage[128] + wage[149] = 0
 Balance[72]: wage[4] + wage[71] - 9 wage[72] + wage[86] + wage[102]
   + wage[143] = 0
 Balance[73]: wage[52] + wage[61] + wage[69] + wage[72] - 9 wage[73]
   + wage[81] + wage[111] + wage[114] + wage[131] + wage[145] = 0
 Balance[74]: wage[1] + wage[16] + wage[27] + wage[54] + wage[60]
   + wage[73] - 9 wage[74] + wage[79] + wage[95] + wage[97] + wage[121]
   + wage[136] = 0
 Balance[75]: wage[47] + wage[57] + wage[61] + wage[74] - 9 wage[75]
   + wage[104] + wage[113] + wage[137] = 0
 Balance[76]: wage[75] - 9 wage[76] + wage[101] = 0
 Balance[77]: wage[75] + wage[76] - 9 wage[77] + wage[86] + wage[89]
   + wage[104] + wage[106] + wage[125] + wage[130] = 0
 Balance[78]: wage[77] - 9 wage[78] + wage[79] + wage[80] + wage[97]
   + wage[98] + wage[149] = 0
 Balance[79]: wage[50] + wage[67] + wage[78] - 9 wage[79] + wage[82]
   + wage[146] = 0
 Balance[80]: wage[6] + wage[10] + wage[20] + wage[53] + wage[79]
   - 8 wage[80] + wage[87] + wage[123] + wage[134] + wage[143] + wage[147]
   = 0
 Balance[81]: wage[8] + wage[24] + wage[29] + wage[71] + wage[79]
   + wage[80] - 9 wage[81] + wage[132] = 0
 Balance[82]: wage[58] + wage[62] + wage[75] + wage[81] - 9 wage[82]
   + wage[93] + wage[106] + wage[146] = 0
 Balance[83]: wage[5] + wage[16] + wage[28] + wage[51] + wage[56]
   + wage[81] + wage[82] - 9 wage[83] + wage[102] + wage[123] = 0
 Balance[84]: wage[48] + wage[73] + wage[75] + wage[78] + wage[83]
   - 9 wage[84] + wage[131] + wage[137] = 0
 Balance[85]: wage[15] + wage[36] + wage[39] + wage[40] + wage[47]
   + wage[57] + wage[62] + wage[73] + wage[84] - 9 wage[85] + wage[91]
   + wage[111] + wage[123] + wage[134] + wage[138] + wage[148] = 0
 Balance[86]: wage[20] + wage[85] - 9 wage[86] + wage[127] + wage[138] = 0
 Balance[87]: wage[1] + wage[2] + wage[9] + wage[86] - 9 wage[87]
   + wage[122] + wage[129] = 0
 Balance[88]: wage[2] + wage[4] + wage[9] + wage[46] + wage[87]
   - 9 wage[88] + 2 wage[107] + wage[147] = 0
 Balance[89]: wage[5] + wage[13] + wage[23] + wage[25] + wage[44]
   + 2 wage[53] + wage[64] + wage[69] + wage[86] + wage[88] - 9 wage[89]
   + wage[100] + wage[115] + wage[119] + wage[121] = 0
 Balance[90]: wage[64] + wage[80] + wage[89] - 9 wage[90] + wage[121] = 0
 Balance[91]: wage[5] + wage[17] + wage[39] + wage[44] + wage[65]
   + wage[69] + wage[76] + wage[85] + wage[90] - 9 wage[91] + wage[117]
   + wage[129] = 0
 Balance[92]: wage[0] + wage[23] + wage[28] + wage[34] + wage[47]
   + wage[53] + wage[84] + wage[91] - 9 wage[92] + wage[105] + wage[128]
   = 0
 Balance[93]: wage[42] + wage[48] + wage[87] + wage[92] - 9 wage[93]
   + wage[95] + wage[96] + wage[98] + wage[103] = 0
 Balance[94]: wage[7] + wage[34] + wage[52] + wage[74] + wage[76]
   + wage[83] + wage[93] - 9 wage[94] + wage[96] + wage[103] + wage[113]
   = 0
 Balance[95]: 2 wage[14] + wage[18] + wage[24] + wage[46] + wage[49]
   + wage[55] + wage[84] + wage[94] - 9 wage[95] + wage[129] = 0
 Balance[96]: wage[22] + wage[32] + wage[95] - 9 wage[96] + wage[108]
   + wage[110] + wage[111] + wage[129] = 0
 Balance[97]: wage[22] + wage[39] + wage[73] + wage[79] + wage[96]
   - 9 wage[97] + wage[136] + wage[139] + wage[141] + wage[144] = 0
 Balance[98]: wage[12] + wage[25] + wage[26] + wage[28] + wage[33]
   + wage[34] + wage[37] + wage[42] + wage[55] + wage[56] + wage[59]
   + wage[74] + wage[88] + wage[97] - 9 wage[98] + wage[133] = 0
 Balance[99]: wage[1] + wage[48] + wage[49] + wage[57] + wage[98]
   - 9 wage[99] + wage[101] + wage[116] + wage[137] = 0
 Balance[100]: wage[12] + wage[19] + wage[41] + wage[48] + wage[51]
   + wage[55] + wage[70] + wage[94] + wage[99] - 9 wage[100] + wage[102]
   + wage[118] + wage[145] = 0
 Balance[101]: wage[69] + wage[74] + wage[100] - 9 wage[101] + wage[144]
   = 0
 Balance[102]: 2 wage[0] + wage[13] + wage[18] + wage[41] + wage[79]
   + wage[101] - 9 wage[102] + wage[126] + wage[135] = 0
 Balance[103]: wage[1] + wage[6] + wage[12] + wage[24] + wage[28]
   + wage[35] + wage[36] + wage[83] + wage[102] - 9 wage[103] + wage[108]
   + 2 wage[109] + wage[114] + wage[115] + wage[116] + wage[123]
   + wage[143] = 0
 Balance[104]: wage[54] + wage[56] + wage[65] + wage[73] + wage[103]
   - 9 wage[104] + wage[105] + wage[123] = 0
 Balance[105]: wage[7] + wage[43] + wage[80] + wage[104] - 9 wage[105]
   + wage[117] + wage[138] = 0
 Balance[106]: wage[0] + wage[87] + wage[90] + wage[105] - 9 wage[106]
   + wage[115] + wage[140] = 0
 Balance[107]: wage[3] + wage[16] + wage[41] + wage[63] + wage[106]
   - 9 wage[107] + wage[122] + wage[134] + wage[144] = 0
 Balance[108]: wage[31] + wage[42] + wage[60] + wage[62] + wage[72]
   + wage[89] + wage[92] + wage[94] + wage[107] - 9 wage[108] + wage[112]
   + wage[136] + wage[143] + wage[147] = 0
 Balance[109]: wage[28] + wage[75] + wage[108] - 9 wage[109] + wage[119]
   + wage[127] = 0
 Balance[110]: wage[5] + wage[8] + wage[32] + wage[58] + wage[82]
   + wage[90] + 3 wage[93] + wage[109] - 8 wage[110] + wage[116]
   + wage[142] = 0
 Balance[111]: wage[25] + wage[28] + wage[57] + wage[78] + wage[95]
   + wage[110] - 9 wage[111] + wage[131] = 0
 Balance[112]: wage[18] + wage[19] + 2 wage[20] + wage[25] + wage[27]
   + wage[55] + wage[56] + wage[65] + wage[76] + wage[88] + wage[93]
   + wage[98] + wage[99] + wage[111] - 9 wage[112] + wage[126] + wage[141]
   = 0
 Balance[113]: wage[36] + wage[43] + wage[54] + wage[112] - 9 wage[113]
   + wage[122] + wage[126] + wage[134] = 0
 Balance[114]: wage[30] + wage[38] + wage[53] + wage[71] + wage[82]
   + 2 wage[90] + 2 wage[113] - 9 wage[114] + wage[119] + wage[125]
   + wage[135] + wage[140] + wage[149] = 0
 Balance[115]: wage[17] + wage[22] + wage[27] + wage[34] + wage[42]
   + wage[114] - 9 wage[115] = 0
 Balance[116]: wage[1] + wage[33] + wage[44] + wage[46] + wage[58]
   + wage[66] + wage[68] + wage[69] + wage[104] + wage[115] - 9 wage[116]
   + wage[120] + wage[138] + wage[149] = 0
 Balance[117]: wage[18] + wage[46] + wage[67] + wage[68] + wage[72]
   + wage[77] + wage[83] + wage[97] + wage[99] + wage[111] + wage[116]
   - 9 wage[117] + wage[123] + wage[147] = 0
 Balance[118]: wage[30] + wage[72] + wage[91] + wage[117] - 9 wage[118]
   + wage[133] = 0
 Balance[119]: wage[27] + wage[43] + wage[54] + wage[104] + wage[116]
   + wage[118] - 9 wage[119] + wage[130] + wage[134] = 0
 Balance[120]: 2 wage[17] + wage[87] + wage[94] + wage[119] - 9 wage[120]
   = 0
 Balance[121]: wage[0] + wage[45] + wage[60] + wage[62] + wage[69]
   + wage[84] + wage[117] + wage[120] - 9 wage[121] + wage[137] + wage[139]
   = 0
 Balance[122]: wage[10] + wage[21] + wage[59] + wage[77] + wage[106]
   + wage[111] + wage[121] - 9 wage[122] + wage[124] = 0
 Balance[123]: wage[8] + wage[14] + wage[22] + wage[33] + wage[43]
   + wage[51] + wage[122] - 9 wage[123] + wage[127] = 0
 Balance[124]: wage[40] + wage[108] + wage[123] - 9 wage[124] + wage[131]
   = 0
 Balance[125]: wage[21] + wage[46] + 2 wage[64] + wage[82] + wage[91]
   + wage[100] + wage[124] - 9 wage[125] = 0
 Balance[126]: wage[36] + wage[45] + wage[65] + wage[88] + wage[117]
   + wage[125] - 9 wage[126] = 0
 Balance[127]: wage[17] + wage[24] + wage[28] + 2 wage[30] + wage[35]
   + wage[50] + wage[58] + wage[121] + wage[126] - 9 wage[127] + wage[134]
   + wage[136] = 0
 Balance[128]: wage[10] + wage[17] + wage[20] + wage[24] + wage[26]
   + wage[45] + wage[46] + wage[74] + wage[126] + wage[127] - 9 wage[128]
   + wage[132] + wage[143] = 0
 Balance[129]: wage[2] + wage[5] + wage[19] + wage[20] + wage[21]
   + wage[36] + wage[53] + wage[89] + wage[101] + wage[102] + wage[113]
   + wage[121] + wage[128] - 9 wage[129] + wage[132] + wage[140] = 0
 Balance[130]: wage[1] + wage[3] + wage[12] + wage[19] + wage[27]
   + wage[65] + wage[96] + wage[127] + wage[129] - 9 wage[130] + wage[144]
   = 0
 Balance[131]: wage[4] + wage[6] + wage[14] + wage[63] + wage[81]
   + 2 wage[83] + wage[107] + wage[126] + wage[130] - 9 wage[131] = 0
 Balance[132]: wage[60] + wage[74] + wage[78] + wage[80] + wage[85]
   + wage[98] + wage[116] + wage[131] - 9 wage[132] = 0
 Balance[133]: wage[6] + wage[16] + wage[66] + wage[71] + wage[97]
   + wage[101] + wage[109] + wage[114] + wage[132] - 9 wage[133]
   + wage[138] = 0
 Balance[134]: wage[4] + wage[11] + wage[30] + wage[37] + wage[54]
   + wage[110] + wage[121] + wage[128] + wage[133] - 9 wage[134]
   + wage[149] = 0
 Balance[135]: 2 wage[10] + wage[30] + wage[62] + wage[70] + wage[98]
   + wage[112] + wage[134] - 9 wage[135] = 0
 Balance[136]: wage[13] + wage[18] + wage[36] + wage[45] + wage[61]
   + wage[96] + wage[104] + wage[117] + wage[120] + wage[133] + wage[135]
   - 9 wage[136] + wage[149] = 0
 Balance[137]: wage[49] + wage[54] + wage[81] + wage[92] + wage[101]
   + wage[136] - 9 wage[137] + wage[139] = 0
 Balance[138]: wage[10] + wage[22] + wage[100] + wage[137] - 9 wage[138]
   = 0
 Balance[139]: wage[26] + wage[45] + wage[51] + wage[57] + wage[59]
   + wage[99] + wage[115] + wage[133] + wage[138] - 9 wage[139] = 0
 Balance[140]: wage[52] + wage[57] + wage[61] + wage[64] + wage[70]
   + wage[112] + wage[115] + wage[124] + wage[139] - 9 wage[140] = 0
 Balance[141]: wage[13] + wage[29] + wage[38] + wage[60] + wage[78]
   + wage[100] + wage[130] + wage[140] - 9 wage[141] = 0
 Balance[142]: wage[14] + wage[39] + wage[68] + wage[103] + wage[109]
   + wage[119] + wage[141] - 9 wage[142] + wage[147] = 0
 Balance[143]: wage[22] + wage[25] + wage[38] + wage[43] + wage[47]
   + wage[48] + wage[58] + wage[59] + wage[68] + wage[83] + wage[125]
   + wage[130] + wage[142] - 9 wage[143] = 0
 Balance[144]: wage[21] + wage[30] + wage[43] + wage[61] + wage[63]
   + wage[70] + wage[75] + wage[78] + wage[81] + wage[92] + wage[109]
   + wage[139] + wage[143] - 9 wage[144] + wage[148] = 0
 Balance[145]: wage[7] + wage[28] + wage[60] + wage[67] + wage[72]
   + wage[93] + wage[107] + wage[136] + wage[144] - 9 wage[145] = 0
 Balance[146]: wage[21] + 2 wage[23] + wage[26] + wage[32] + wage[43]
   + wage[49] + wage[57] + wage[67] + wage[84] + 2 wage[131] + wage[145]
   - 9 wage[146] = 0
 Balance[147]: 2 wage[23] + wage[27] + wage[29] + wage[65] + wage[114]
   + wage[123] + wage[132] + wage[142] + wage[146] - 9 wage[147] = 0
 Balance[148]: wage[66] + wage[74] + wage[85] + wage[118] + wage[120]
   + wage[141] + wage[146] + wage[147] - 9 wage[148] = 0
 Balance[149]: wage[1] + wage[40] + wage[95] + wage[105] + wage[122]
   + wage[142] + wage[148] - 9 wage[149] = 0
Bounds
End
