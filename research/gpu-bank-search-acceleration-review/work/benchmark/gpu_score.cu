// The rightmost option varies fastest, matching the reference Cartesian order.
extern "C" __global__ void score_ordinals(
    unsigned long long first, int count, int motifs, int chart_count,
    const unsigned long long *prefix, const int *meta,
    const long long *baseline, const int *options, const long long *exterior,
    const long long *tariff, unsigned int *hits, long long *boundary,
    int audit, int *all_charts, long long *all_scores)
{
    const unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= (unsigned int)count) return;
    const unsigned long long global_row = first + row;
    int low = 0, high = motifs;
    while (high - low > 1) {
        const int middle = (low + high) / 2;
        if (prefix[middle] <= global_row) low = middle;
        else high = middle;
    }
    const int m = low;
    const int *d = meta + 19 * m;
    const unsigned long long local = global_row - prefix[m];
    unsigned long long ordinal = local;
    int chosen[4];
    int changes = 0;
    long long value = -baseline[m];
    #pragma unroll
    for (int j = 3; j >= 0; --j) {
        const unsigned int radix = d[j];
        const int position = d[4+j] + (unsigned int)(ordinal % radix);
        ordinal /= radix;
        chosen[j] = options[position];
        changes += chosen[j] != d[8+j];
        value += exterior[position];
        if (audit) all_charts[4*row+j] = chosen[j];
    }
    const long long invalid = -100000000000000LL;
    bool legal = changes >= 3;
    int k = 0;
    #pragma unroll
    for (int u = 0; u < 4; ++u) {
        #pragma unroll
        for (int v = u+1; v < 4; ++v) {
            const int weight = d[12+k++];
            if (weight) {
                const long long term = tariff[chosen[u]*chart_count + chosen[v]];
                if (term == invalid) legal = false;
                else value += term * weight;
            }
        }
    }
    const long long score = legal ? value : invalid;
    if (audit) all_scores[row] = score;
    if (score >= -(long long)d[18]-2) {
        const unsigned int i = atomicAdd(hits, 1u);
        boundary[3*i] = m;
        boundary[3*i+1] = (long long)local;
        boundary[3*i+2] = score;
    }
}
