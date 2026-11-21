import math
import random

import pytest

import curvekit
from curvekit import core


# --------------------------------------------------------------------------
# 独立校验工具：不依赖被测实现的距离定义
# --------------------------------------------------------------------------
def _seg_dist2(px, py, ax, ay, bx, by):
    dx = bx - ax
    dy = by - ay
    l2 = dx * dx + dy * dy
    if l2 == 0.0:
        ex = px - ax
        ey = py - ay
        return ex * ex + ey * ey
    t = ((px - ax) * dx + (py - ay) * dy) / l2
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    ex = px - (ax + t * dx)
    ey = py - (ay + t * dy)
    return ex * ex + ey * ey


def _max_deviation(src, out):
    """简化结果相对原始折线的最大双向偏差（点到输出折线的最短距离）。"""
    worst = 0.0
    for px, py in src:
        best = math.inf
        for (ax, ay), (bx, by) in zip(out, out[1:]):
            d = _seg_dist2(px, py, ax, ay, bx, by)
            if d < best:
                best = d
        if best > worst:
            worst = best
    return math.sqrt(worst)


def _is_subsequence(sub, full):
    it = iter(full)
    return all(any(x == y for y in it) for x in sub)


def _pairs(seq):
    return list(seq)


def _zigzag(n, amp=1.0):
    return [(float(i), amp * float(i & 1)) for i in range(n)]


# --------------------------------------------------------------------------
# 统计结构性质
# --------------------------------------------------------------------------
@pytest.mark.parametrize("n", [1, 2, 3, 10, 100, 1000])
@pytest.mark.parametrize("tol", [0.0, 1e-9, 0.1, 0.5, 2.0, 1e9])
def test_endpoints_and_subsequence(n, tol):
    pts = _zigzag(n)
    out = curvekit.Simplifier(tol).simplify(pts)
    assert out[0] == pts[0] and out[-1] == pts[-1]
    assert _is_subsequence(out, pts)
    assert len(out) <= n


@pytest.mark.parametrize("seed", range(20))
def test_max_deviation_bound(seed):
    rng = random.Random(seed)
    n = rng.randint(3, 120)
    pts = []
    x = y = 0.0
    for _ in range(n):
        if rng.random() < 0.15:  # 注入重复点
            pts.append((x, y))
            continue
        x += rng.uniform(-3, 3)
        y += rng.uniform(-3, 3)
        pts.append((x, y))
    for tol in (0.05, 0.3, 1.5):
        out = curvekit.Simplifier(tol).simplify(pts)
        assert _max_deviation(pts, out) <= tol + 1e-9


@pytest.mark.parametrize("seed", range(20))
def test_monotonic_counts(seed):
    rng = random.Random(1000 + seed)
    n = rng.randint(4, 150)
    pts = [(rng.uniform(-5, 5), rng.uniform(-5, 5)) for _ in range(n)]
    tols = sorted(rng.uniform(0.0, 8.0) for _ in range(7))
    counts = [len(curvekit.Simplifier(t).simplify(pts)) for t in tols]
    for a, b in zip(counts, counts[1:]):
        assert b <= a, (tols, counts)


def test_tolerance_zero_identity():
    for pts in (_zigzag(50), [(0.5, 0.5)], [], [(1.0, 1.0), (2.0, 2.0)]):
        out = curvekit.Simplifier(0.0).simplify(pts)
        assert out == pts
        assert out is not pts


# --------------------------------------------------------------------------
# 大规模 / 栈深 / 性能
# --------------------------------------------------------------------------
@pytest.mark.parametrize("shape", ["line", "sine", "random", "zigzag"])
def test_large_100k_time(shape):
    import time

    n = 100_000
    if shape == "line":
        pts = [(float(i), 0.0) for i in range(n)]
    elif shape == "sine":
        pts = [(i * 0.01, math.sin(i * 0.01)) for i in range(n)]
    elif shape == "zigzag":
        pts = _zigzag(n)
    else:
        rng = random.Random(9)
        pts = []
        x = y = 0.0
        for _ in range(n):
            x += rng.uniform(-1, 1)
            y += rng.uniform(-1, 1)
            pts.append((x, y))
    tol = 0.05
    t0 = time.perf_counter()
    out = curvekit.Simplifier(tol).simplify(pts)
    elapsed = time.perf_counter() - t0
    assert 2 <= len(out) <= n
    assert elapsed < 0.4, f"{shape}: {elapsed * 1000:.1f} ms"


def test_deep_input_no_recursion():
    # 逐层只分裂出一个点、形成极深的二分链：递归实现会在此爆栈。
    n = 20_000
    pts = []
    for i in range(n):
        # 每个点的偏差随下标单调上升，迫使分裂点总是靠近一端
        pts.append((float(i), float(i) * 1e-4 if i % 2 else 0.0))
    out = curvekit.Simplifier(1e-9).simplify(pts)
    assert out[0] == pts[0] and out[-1] == pts[-1]
    assert 2 <= len(out) <= n


def test_wide_input_no_recursion():
    import sys

    n = 200_000
    old = sys.getrecursionlimit()
    sys.setrecursionlimit(60)  # 递归实现会立刻 RecursionError
    try:
        pts = _zigzag(n)
        out = curvekit.Simplifier(0.1).simplify(pts)
        assert len(out) == n
    finally:
        sys.setrecursionlimit(old)


# --------------------------------------------------------------------------
# 并发 / 幂等 / 不修改输入
# --------------------------------------------------------------------------
def test_concurrent_same_result():
    import concurrent.futures

    pts = [(i * 0.01, math.sin(i * 0.01)) for i in range(20_000)]
    expected = curvekit.Simplifier(0.02).simplify(pts)
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        futures = [
            ex.submit(curvekit.Simplifier(0.02).simplify, pts) for _ in range(16)
        ]
        results = [f.result() for f in futures]
    for r in results:
        assert r == expected
    # 共享同一个实例也应安全
    s = curvekit.Simplifier(0.02)
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        assert all(r == expected for r in ex.map(s.simplify, [pts] * 8))


def test_no_mutation_and_fresh_list():
    src = _zigzag(500)
    snapshot = list(src)
    out = curvekit.Simplifier(0.2).simplify(src)
    assert src == snapshot
    assert out is not src
    out.append((123.0, 456.0))
    assert src == snapshot


def test_iterator_and_tuple_input():
    pts = _zigzag(30)
    assert curvekit.Simplifier(0.1).simplify(iter(pts)) == curvekit.Simplifier(
        0.1
    ).simplify(tuple(pts))


def test_numpy_input_if_available():
    np = pytest.importorskip("numpy")
    pts = np.array([(i * 0.1, math.sin(i * 0.1)) for i in range(1000)])
    before = pts.copy()
    out = curvekit.Simplifier(0.01).simplify(pts)
    assert isinstance(out, list) and all(isinstance(p, tuple) for p in out)
    assert np.array_equal(pts, before)
    # 与等价的 tuple 列表输入结果一致
    assert out == curvekit.Simplifier(0.01).simplify([tuple(map(float, r)) for r in pts])


def test_numpy_shape_errors_if_available():
    np = pytest.importorskip("numpy")
    for bad in (np.array([1.0, 2.0]), np.array([[1.0, 2.0, 3.0]]), np.array(5.0)):
        with pytest.raises((ValueError, TypeError)):
            curvekit.Simplifier(0.1).simplify(bad)
    assert curvekit.Simplifier(0.1).simplify(np.empty((0, 2))) == []


# --------------------------------------------------------------------------
# 非法输入
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "bad", [-1.0, -1e-12, math.nan, math.inf, -math.inf, "0.1", None, [1.0], 1j]
)
def test_bad_tolerance_rejected(bad):
    with pytest.raises((ValueError, TypeError)):
        curvekit.Simplifier(bad)


@pytest.mark.parametrize(
    "coord",
    [
        math.nan,
        math.inf,
        -math.inf,
    ],
)
def test_non_finite_coords_rejected(coord):
    with pytest.raises(ValueError):
        curvekit.Simplifier(0.1).simplify([(0.0, 0.0), (coord, 1.0), (2.0, 2.0)])
    with pytest.raises(ValueError):
        curvekit.Simplifier(0.1).simplify([(0.0, 0.0), (1.0, coord)])


@pytest.mark.parametrize(
    "bad_points",
    [
        [(0.0, 0.0), (1.0,)],  # 维度不对
        [(0.0, 0.0, 0.0)],  # 三元组
        [(0.0, "x")],  # 非数值
        [None],
        "not-points",
    ],
)
def test_malformed_points_rejected(bad_points):
    with pytest.raises((ValueError, TypeError)):
        curvekit.Simplifier(0.1).simplify(bad_points)


# --------------------------------------------------------------------------
# 正确性对照与内部一致性
# --------------------------------------------------------------------------
def _reference_dp(pts, tol):
    """参考实现（与默认分裂规则相同，用于小规模逐点对照）。"""
    if len(pts) <= 2:
        return list(pts)
    ax, ay = pts[0]
    bx, by = pts[-1]
    best = -1.0
    idx = 0
    for k in range(1, len(pts) - 1):
        d = _seg_dist2(pts[k][0], pts[k][1], ax, ay, bx, by)
        if d > best:
            best = d
            idx = k
    if math.sqrt(best) > tol:
        span = len(pts) - 1
        if idx <= span // 8 or span - idx <= span // 8:
            idx = span // 2
        left = _reference_dp(pts[: idx + 1], tol)
        right = _reference_dp(pts[idx:], tol)
        return left[:-1] + right
    return [pts[0], pts[-1]]


@pytest.mark.parametrize("seed", range(30))
def test_matches_reference(seed):
    rng = random.Random(500 + seed)
    n = rng.randint(3, 80)
    pts = [(rng.uniform(-4, 4), rng.uniform(-4, 4)) for _ in range(n)]
    tol = rng.choice((0.1, 0.5, 1.0, 2.0))
    assert curvekit.Simplifier(tol).simplify(pts) == _reference_dp(pts, tol)


@pytest.mark.parametrize("seed", range(10))
def test_repeatable(seed):
    rng = random.Random(900 + seed)
    n = rng.randint(40, 300)
    pts = []
    x = y = 0.0
    for _ in range(n):
        x += rng.uniform(-2, 2)
        y += rng.uniform(-2, 2)
        pts.append((x, y))
    tol = rng.choice((0.05, 0.3, 1.0))
    first = curvekit.Simplifier(tol).simplify(pts)
    for _ in range(5):
        assert curvekit.Simplifier(tol).simplify(pts) == first


def test_tolerance_extremes():
    pts = _zigzag(400)
    assert len(curvekit.Simplifier(1e12).simplify(pts)) == 2
    huge = [(1e308, -1e308), (0.0, 1e308), (-1e308, -1e308)]
    out = curvekit.Simplifier(1e30).simplify(huge)
    assert out[0] == huge[0] and out[-1] == huge[-1]


def test_duplicate_points_heavy():
    pts = [(0.0, 0.0)] * 50 + [(1.0, 0.0)] * 50 + [(1.0, 1.0)] * 50
    out = curvekit.Simplifier(0.1).simplify(pts)
    # (1, 0) 是真实的拐角，必须保留；重复点被折叠
    assert out == [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)]
    assert curvekit.Simplifier(0.0).simplify(pts) == pts
