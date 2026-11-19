"""Build the original ranked motif plan using round-local exterior tariff caches."""

from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations

import numpy as np

from bank_graph import INVALID, component_graph, incidence, price_tables


@dataclass
class MotifPlan:
    table: np.ndarray
    old: np.ndarray
    neighbors: list
    selected: list
    groups: list
    census: Counter
    nonexcluded: int
    stats: dict


def connected_motifs(free, adjacency):
    """Enumerate each connected triple and quadruple once as in the reference."""
    triples = set()
    for u in free:
        for v, w in combinations(adjacency[u], 2):
            triples.add(tuple(sorted((u, v, w))))
    yield from triples

    quads = set()
    for triple in triples:
        extensions = set().union(*(adjacency[u] for u in triple)).difference(triple)
        for v in extensions:
            quads.add(tuple(sorted((*triple, v))))
    yield from quads


class RankedExterior:
    """Share ranked options, including the reference's optional old-chart tail."""

    def __init__(self, valid_ids, scores, limit):
        order = np.argsort(-scores[valid_ids], kind="stable")
        ranked = valid_ids[order]
        stop = limit or len(ranked)
        self.basic = ranked[:stop], scores[ranked[:stop]]
        self.maximum = int(self.basic[1][0])
        self.tail_prices = {int(v): int(scores[v]) for v in ranked[stop:]}
        self.variants = {}

    def options(self, bank_chart, cluster_charts):
        if not self.tail_prices:
            return self.basic
        # The baseline checks the bank's own chart before the ordered cluster.
        tail = tuple(dict.fromkeys(v for v in (bank_chart, *cluster_charts)
                                  if v in self.tail_prices))
        if not tail:
            return self.basic
        if tail not in self.variants:
            charts, values = self.basic
            self.variants[tail] = (
                np.concatenate((charts, np.array(tail, dtype=np.int64))),
                np.concatenate((values, np.array([self.tail_prices[v] for v in tail],
                                                 dtype=np.int64))),
            )
        return self.variants[tail]


def select_motifs(graph, layout, kappa, motif_limit=10000, option_limit=24, *,
                  min_change=3, family="legacy"):
    """Return ordered legacy (upper, cluster, info) triples for this input round.

    Interior paid terms are at most zero, so the sum of exterior maxima minus
    the old incident price bounds the rounded score from above. Each old/new
    price pair contributes at most one unit of total rounding error per edge.
    We reuse the reference upper >= -edgecount filter, then sort by (-upper,
    cluster). The motif and option limits still restrict the search space.
    """
    if family != "legacy" or min_change != 3:
        raise ValueError("legacy preparation supports connected 3/4-bank motifs and min_change=3")
    if option_limit < 0 or motif_limit < 0:
        raise ValueError("motif and option limits must be nonnegative")

    groups, fixed, old, edges = component_graph(graph, layout)
    neighbors = incidence(old, edges)
    tables, _ = price_tables(graph['frames'], kappa)
    table = tables[0]
    # Contiguous chart columns are repeatedly accessed while pricing exteriors.
    by_chart = np.ascontiguousarray(table.T)
    old_ids = old.tolist()
    free = set(range(len(groups))) - fixed
    multiplicities = [Counter(v for kind, _, v, _ in rows) for rows in neighbors]
    if any(kind != 0 for kind, _, _, _ in edges):
        raise ValueError("the public-boundary search requires paid edges only")
    adjacency = {u: free.intersection(multiplicities[u]) for u in free}
    chart_counts = {u: Counter(chart for _, chart, _, _ in neighbors[u]) for u in free}
    old_prices = {u: sum(n * int(table[old_ids[u], chart])
                         for chart, n in chart_counts[u].items()) for u in free}

    @lru_cache(maxsize=None)
    def ranked_exterior(signature):
        # Equal exterior chart multisets imply exactly equal legality and price,
        # independently of the identities and ordering of the incident edges.
        values = np.zeros(len(table), dtype=np.int64)
        valid = np.ones(len(table), dtype=bool)
        for chart, count in signature:
            column = by_chart[chart]
            valid &= column != INVALID
            values += column * count
        return RankedExterior(np.flatnonzero(valid), values, option_limit)

    @lru_cache(maxsize=None)
    def bank_terms(u, interior):
        counts = chart_counts[u].copy()
        price = old_prices[u]
        edgecount = len(neighbors[u])
        for v in interior:
            chart = old_ids[v]
            count = multiplicities[u][v]
            counts[chart] -= count
            # Charge the smaller bank id for each shared interior edge.
            if u > v:
                price -= count * int(table[old_ids[u], chart])
                edgecount -= count
        signature = tuple(sorted((chart, n) for chart, n in counts.items() if n))
        return ranked_exterior(signature), price, edgecount

    records = []
    census = Counter()
    for cluster in connected_motifs(free, adjacency):
        census[len(cluster)] += 1
        rows = []
        maximum = baseline = edgecount = 0
        for u in cluster:
            adjacent = adjacency[u]
            interior = tuple(v for v in cluster if v in adjacent)
            row, price, count = bank_terms(u, interior)
            rows.append(row)
            maximum += row.maximum
            baseline += price
            edgecount += count
        upper = maximum - baseline
        if upper >= -edgecount:
            records.append((upper, cluster, baseline, edgecount, rows))
    records.sort(key=lambda r: (-r[0], r[1]))
    nonexcluded = len(records)
    selected = []
    for upper, cluster, baseline, edgecount, rows in records[:motif_limit]:
        cluster_charts = tuple(old_ids[u] for u in cluster)
        options, exterior = [], []
        for chart, row in zip(cluster_charts, rows):
            charts, values = row.options(chart, cluster_charts)
            options.append(charts)
            exterior.append(values)
        internal = Counter({(i, j): multiplicities[u][v]
                            for (i, u), (j, v) in combinations(enumerate(cluster), 2)
                            if v in multiplicities[u]})
        info = options, exterior, internal, baseline, edgecount, upper
        selected.append((upper, cluster, info))

    stats = {"bank_exterior_cache_hits": bank_terms.cache_info().hits,
             "bank_exterior_cache_misses": bank_terms.cache_info().misses,
             "chart_multiset_cache_hits": ranked_exterior.cache_info().hits,
             "chart_multiset_cache_misses": ranked_exterior.cache_info().misses,
             "cache_lifetime": "one graph/pricing round, no persistent search cache"}
    return MotifPlan(table, old, neighbors, selected, groups, census, nonexcluded, stats)
