## 数据切分
输入的张量x形状为[N, D]， tile切分应该是按行划分的

输出张量y ：[1, N(N-1)/2]，其中y[k] = Pdistance( x[i], x[j] ), i < j