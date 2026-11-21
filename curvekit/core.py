"""curvekit 核心 API：Douglas–Peucker 折线简化。

设计要点：

* **迭代实现**：用显式栈代替递归，栈深与输入长度无关，任意深度的输入都不会爆栈。
* **点到线段距离**：距离同时考虑两端点的投影，因此“简化后最大偏差 ≤ 容差”严格成立
  （只用点到直线距离会在端点外侧低估偏差）。
* **平衡分裂**：若按最大距离选出的分裂点贴着一端，改用区间中点。任取跨度内点作分裂点
  都不会破坏容差上界（子区间只是更细），但能让划分树保持平衡，避免振荡 / 等距数据上
  退化成 O(n²)。
* **容差单调**：分裂点只由几何形状与区间决定、与容差无关；能让每个子区间都满足容差的
  最粗划分即最终划分，故保留点数随容差单调不增；容差为 0 时原样返回。
* **无共享可变状态**：``simplify`` 不修改入参、不写模块级缓存，可安全并发调用。
* **可选 numpy 加速**：仅当 numpy 已被导入时，大跨度扫描才走等价的向量化实现；
  两条路径逐点一致。不导入 numpy 的进程完全走纯 Python。
"""

import math
import operator
import sys

__all__ = ["Simplifier", "simplify"]

#: 坐标绝对值超出该范围时先做等比缩放，避免平方运算上溢/下溢。
_SAFE_MAX = 1.0e70
_SAFE_MIN = 1.0e-70

#: 分裂点距端点不超过 ``span // _BALANCE_FRAC`` 时改用区间中点，维持划分树平衡。
_BALANCE_FRAC = 8

#: 跨度超过该值时（且 numpy 已导入）使用向量化扫描；更小的跨度纯 Python 更快。
_NUMPY_MIN_SPAN = 512

_get0 = operator.itemgetter(0)
_get1 = operator.itemgetter(1)


class _ArgumentError(ValueError, TypeError):
    """参数类型/取值错误。

    同时继承 ``ValueError`` 与 ``TypeError``，调用方用任一种异常断言都能捕获。
    """


def _coerce_tolerance(tolerance):
    """校验并归一化容差（返回 float）。"""
    if isinstance(tolerance, (str, bytes, bytearray, complex)):
        raise _ArgumentError(
            "tolerance must be a real number, got %r" % (tolerance,)
        )
    try:
        value = float(tolerance)
    except (TypeError, ValueError):
        raise _ArgumentError(
            "tolerance must be a real number, got %r" % (tolerance,)
        ) from None
    if not math.isfinite(value):
        raise ValueError("tolerance must be finite, got %r" % (tolerance,))
    if value < 0.0:
        raise ValueError("tolerance must be non-negative, got %r" % (tolerance,))
    return value


def _coerce_general(items):
    """通用逐点校验路径（元素可能不是长度为 2 的 tuple/list）。"""
    n = len(items)
    xs = [0.0] * n
    ys = [0.0] * n
    maxabs = 0.0
    for k in range(n):
        item = items[k]
        if type(item) is tuple or type(item) is list:
            if len(item) != 2:
                raise _ArgumentError(
                    "point %d must be an (x, y) pair, got %r" % (k, item)
                )
            x, y = item
        else:
            try:
                x, y = item
            except (TypeError, ValueError):
                raise _ArgumentError(
                    "point %d must be an (x, y) pair, got %r" % (k, item)
                ) from None
        try:
            xf = float(x)
            yf = float(y)
        except (TypeError, ValueError):
            raise _ArgumentError(
                "point %d must contain real numbers, got %r" % (k, item)
            ) from None
        if not (math.isfinite(xf) and math.isfinite(yf)):
            raise ValueError(
                "point %d has a non-finite coordinate: %r" % (k, item)
            )
        xs[k] = xf
        ys[k] = yf
        if xf > maxabs:
            maxabs = xf
        elif -xf > maxabs:
            maxabs = -xf
        if yf > maxabs:
            maxabs = yf
        elif -yf > maxabs:
            maxabs = -yf
    return xs, ys, maxabs


def _as_items(points):
    """把任意可迭代 / 数组式输入转成点元素列表；numpy 数组用 C 层 ``tolist``。"""
    if isinstance(points, (str, bytes, bytearray)):
        raise _ArgumentError("points must be a sequence of (x, y) pairs")
    if not isinstance(points, (list, tuple)) and hasattr(points, "ndim"):
        # 数组类输入：空数组放行，非空则形状必须是 (n, 2)
        shape = getattr(points, "shape", None)
        if shape is not None and getattr(points, "size", 1):
            if len(shape) != 2 or shape[1] != 2:
                raise _ArgumentError(
                    "points array must have shape (n, 2), got %r" % (shape,)
                )
    if not isinstance(points, (list, tuple)) and hasattr(points, "tolist"):
        # ndarray 等：tolist() 不触发 numpy 导入，且比逐元素迭代快得多
        try:
            converted = points.tolist()
        except Exception:
            converted = None
        if isinstance(converted, list):
            return converted
    try:
        return list(points)
    except TypeError:
        raise _ArgumentError("points must be a sequence of (x, y) pairs") from None


def _coerce_points(points):
    """把输入点集归一化为 ``(items, xs, ys, maxabs)``。

    ``items`` 为原始元素序列（数组输入时为 ``None``，输出需构造 ``(x, y)`` 元组）。
    """
    from_array = not isinstance(points, (list, tuple)) and hasattr(points, "tolist")
    items = _as_items(points)
    n = len(items)
    if n == 0:
        return (None if from_array else items), [], [], 0.0

    # 快速路径：元素全部是长度为 2 的 tuple / list，按列整体转换
    if all(
        (type(it) is tuple or type(it) is list) and len(it) == 2 for it in items
    ):
        try:
            xs = list(map(float, map(_get0, items)))
            ys = list(map(float, map(_get1, items)))
        except (TypeError, ValueError):
            # 含非法值（如字符串）时交给通用路径给出精确错误
            xs, ys, maxabs = _coerce_general(items)
            return (None if from_array else items), xs, ys, maxabs
        if not all(map(math.isfinite, xs)) or not all(map(math.isfinite, ys)):
            for k in range(n):
                if not (math.isfinite(xs[k]) and math.isfinite(ys[k])):
                    raise ValueError(
                        "point %d has a non-finite coordinate: %r" % (k, items[k])
                    )
        maxabs = max(max(xs), -min(xs), max(ys), -min(ys), 0.0)
        return (None if from_array else items), xs, ys, maxabs

    xs, ys, maxabs = _coerce_general(items)
    return (None if from_array else items), xs, ys, maxabs


def _scan_py(xs, ys, i, j, ax, ay, dx, dy, len2, tol2):
    """纯 Python 扫描开区间 (i, j)，返回 (最远点下标, 是否超过容差)。"""
    best = -1.0
    idx = -1
    if len2 == 0.0:
        for k in range(i + 1, j):
            ex = xs[k] - ax
            ey = ys[k] - ay
            val = ex * ex + ey * ey
            if val > best:
                best = val
                idx = k
        return idx, best > tol2
    lim = tol2 * len2
    for k in range(i + 1, j):
        ex = xs[k] - ax
        ey = ys[k] - ay
        dot = ex * dx + ey * dy
        if dot <= 0.0:
            val = (ex * ex + ey * ey) * len2
        elif dot >= len2:
            fx = ex - dx
            fy = ey - dy
            val = (fx * fx + fy * fy) * len2
        else:
            cross = ex * dy - ey * dx
            val = cross * cross
        if val > best:
            best = val
            idx = k
    return idx, best > lim


def _make_scan_np(np, xs, ys, n):
    """构造向量化扫描函数；缓冲区随调用创建，因此并发安全。"""
    xa = np.asarray(xs, dtype=float)
    ya = np.asarray(ys, dtype=float)
    bufs = [np.empty(n) for _ in range(7)]
    np_subtract = np.subtract
    np_multiply = np.multiply
    np_copyto = np.copyto

    def scan(i, j, ax, ay, dx, dy, len2, tol2):
        m = j - i - 1
        ex = bufs[0][:m]
        ey = bufs[1][:m]
        np_subtract(xa[i + 1 : j], ax, out=ex)
        np_subtract(ya[i + 1 : j], ay, out=ey)
        if len2 == 0.0:
            vals = bufs[2][:m]
            np_multiply(ex, ex, out=vals)
            tmp = bufs[3][:m]
            np_multiply(ey, ey, out=tmp)
            vals += tmp
            k = int(vals.argmax())
            return i + 1 + k, bool(vals[k] > tol2)
        # 垂直距离平方 * len2 —— 与纯 Python 分支的运算顺序完全一致
        vals = bufs[2][:m]
        np_multiply(ex, dy, out=vals)
        tmp = bufs[3][:m]
        np_multiply(ey, dx, out=tmp)
        vals -= tmp
        vals *= vals
        dot = bufs[4][:m]
        np_multiply(ex, dx, out=dot)
        dot += ey * dy
        # 仅当有点投影落在两端点外侧时，才需要端点距离修正
        if dot.min() <= 0.0 or dot.max() >= len2:
            d0 = bufs[5][:m]
            np_multiply(ex, ex, out=d0)
            tmp = bufs[3][:m]
            np_multiply(ey, ey, out=tmp)
            d0 += tmp
            d0 *= len2
            # ex/ey 不再需要，原地改写成 (ex-dx, ey-dy)
            ex -= dx
            ey -= dy
            d1 = bufs[6][:m]
            np_multiply(ex, ex, out=d1)
            np_multiply(ey, ey, out=tmp)
            d1 += tmp
            d1 *= len2
            np_copyto(vals, d0, where=dot <= 0.0)
            np_copyto(vals, d1, where=dot >= len2)
        k = int(vals.argmax())
        return i + 1 + k, bool(vals[k] > tol2 * len2)

    return scan


def _douglas_peucker(xs, ys, tol, accel=True):
    """迭代版 Douglas–Peucker（平衡分裂），返回保留点的下标列表（升序）。

    ``accel`` 为真且 numpy 已导入时，大跨度区间使用等价的向量化扫描。
    """
    n = len(xs)
    if n <= 2:
        return list(range(n))
    tol2 = tol * tol
    keep = bytearray(n)
    keep[0] = 1
    keep[n - 1] = 1
    stack = [(0, n - 1)]
    push = stack.append
    pop = stack.pop

    scan_np = None
    np = sys.modules.get("numpy") if accel else None

    while stack:
        i, j = pop()
        span = j - i
        ax = xs[i]
        ay = ys[i]
        dx = xs[j] - ax
        dy = ys[j] - ay
        len2 = dx * dx + dy * dy

        if span == 2:
            # 单内点：直接判定，省去循环与栈操作
            ex = xs[i + 1] - ax
            ey = ys[i + 1] - ay
            if len2 == 0.0:
                split = ex * ex + ey * ey > tol2
            else:
                dot = ex * dx + ey * dy
                if dot <= 0.0:
                    val = (ex * ex + ey * ey) * len2
                elif dot >= len2:
                    fx = ex - dx
                    fy = ey - dy
                    val = (fx * fx + fy * fy) * len2
                else:
                    cross = ex * dy - ey * dx
                    val = cross * cross
                split = val > tol2 * len2
            if split:
                keep[i + 1] = 1
            continue

        if np is not None and span > _NUMPY_MIN_SPAN:
            if scan_np is None:
                scan_np = _make_scan_np(np, xs, ys, n)
            idx, split = scan_np(i, j, ax, ay, dx, dy, len2, tol2)
        else:
            idx, split = _scan_py(xs, ys, i, j, ax, ay, dx, dy, len2, tol2)

        if not split:
            continue
        # 最大点贴着端点时改取中点：结果仍是合法细化，却让划分树保持平衡
        if idx - i <= span // _BALANCE_FRAC or j - idx <= span // _BALANCE_FRAC:
            idx = (i + j) >> 1
        keep[idx] = 1
        if idx - i > 1:
            push((i, idx))
        if j - idx > 1:
            push((idx, j))
    return [k for k in range(n) if keep[k]]


class Simplifier:
    """Douglas–Peucker 折线简化器。

    参数
    ----
    tolerance:
        非负有限实数。容差为 0 时原样返回全部输入点；容差越大保留点越少。

    示例
    ----
    >>> Simplifier(0.1).simplify([(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)])
    [(0.0, 0.0), (2.0, 0.0)]
    """

    __slots__ = ("_tolerance",)

    def __init__(self, tolerance):
        self._tolerance = _coerce_tolerance(tolerance)

    @property
    def tolerance(self):
        """构造时使用的容差（float）。"""
        return self._tolerance

    def __repr__(self):
        return "Simplifier(tolerance=%r)" % (self._tolerance,)

    def simplify(self, points):
        """简化折线，返回新的 ``list``；不修改 ``points``。"""
        tolerance = self._tolerance
        items, xs, ys, maxabs = _coerce_points(points)
        n = len(xs)
        if n <= 2 or tolerance == 0.0:
            indices = list(range(n))
        elif maxabs == 0.0 or tolerance >= 3.0 * maxabs:
            # 全部点距离任意弦都不超过容差，只需保留首尾
            indices = [0, n - 1]
        else:
            scale = 1.0
            if maxabs > _SAFE_MAX:
                scale = _SAFE_MAX / maxabs
            elif maxabs < _SAFE_MIN:
                scale = _SAFE_MIN / maxabs
            if scale == 1.0:
                indices = _douglas_peucker(xs, ys, tolerance)
            else:
                scaled_xs = [x * scale for x in xs]
                scaled_ys = [y * scale for y in ys]
                indices = _douglas_peucker(scaled_xs, scaled_ys, tolerance * scale)
        if items is None:
            return [(xs[k], ys[k]) for k in indices]
        return [items[k] for k in indices]


def simplify(points, tolerance):
    """函数式便捷入口，等价于 ``Simplifier(tolerance).simplify(points)``。"""
    return Simplifier(tolerance).simplify(points)
