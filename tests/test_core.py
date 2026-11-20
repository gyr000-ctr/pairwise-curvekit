import math

import pytest

import curvekit

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

LINE = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0)]
ZIG = [(0.0, 0.0), (1.0, 1.0), (2.0, 0.0), (3.0, 1.0)]


def _check_simp_endpoints():
    out = curvekit.Simplifier(0.1).simplify(ZIG)
    return out[0] == ZIG[0] and out[-1] == ZIG[-1]


def _check_simp_line():
    return curvekit.Simplifier(0.1).simplify(LINE) == [LINE[0], LINE[-1]]


def _check_simp_zig():
    return len(curvekit.Simplifier(0.1).simplify(ZIG)) == 4


def _check_simp_zero():
    return curvekit.Simplifier(0.0).simplify(ZIG) == ZIG


def _check_simp_monotonic():
    a = len(curvekit.Simplifier(0.1).simplify(ZIG))
    b = len(curvekit.Simplifier(10.0).simplify(ZIG))
    return b <= a


def _check_simp_error_bound():
    out = curvekit.Simplifier(10.0).simplify(ZIG)
    return len(out) == 2


def _check_simp_bad():
    return _raises(lambda: curvekit.Simplifier(-1.0))


def _check_simp_empty():
    return curvekit.Simplifier(0.1).simplify([]) == []


def _check_simp_single():
    return curvekit.Simplifier(0.1).simplify([(1.0, 1.0)]) == [(1.0, 1.0)]


def _check_simp_det():
    return curvekit.Simplifier(0.5).simplify(ZIG) == curvekit.Simplifier(0.5).simplify(ZIG)


def _check_simp_type():
    return isinstance(curvekit.Simplifier(0.1).simplify(ZIG), list)


def _check_simp_two():
    return curvekit.Simplifier(0.1).simplify([(0.0, 0.0), (1.0, 1.0)]) == [(0.0, 0.0), (1.0, 1.0)]


def _check_simp_large():
    pts = [(float(i), 0.0) for i in range(100)]
    return len(curvekit.Simplifier(0.1).simplify(pts)) == 2


def _check_simp_dup_points():
    pts = [(0.0, 0.0), (0.0, 0.0), (1.0, 0.0)]
    return len(curvekit.Simplifier(0.1).simplify(pts)) == 2


def _check_simp_no_mutation():
    src = list(ZIG)
    curvekit.Simplifier(0.1).simplify(ZIG)
    return ZIG == src

CASES = [("simp_endpoints", "_check_simp_endpoints"), ("simp_line", "_check_simp_line"), ("simp_zig", "_check_simp_zig"), ("simp_zero", "_check_simp_zero"), ("simp_monotonic", "_check_simp_monotonic"), ("simp_error_bound", "_check_simp_error_bound"), ("simp_bad", "_check_simp_bad"), ("simp_empty", "_check_simp_empty"), ("simp_single", "_check_simp_single"), ("simp_det", "_check_simp_det"), ("simp_type", "_check_simp_type"), ("simp_two", "_check_simp_two"), ("simp_large", "_check_simp_large"), ("simp_dup_points", "_check_simp_dup_points"), ("simp_no_mutation", "_check_simp_no_mutation")]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert globals()[check]() is True
