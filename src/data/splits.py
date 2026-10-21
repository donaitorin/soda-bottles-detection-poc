"""Store-level train / val / test assignment and the checks that guard it."""
import random
import re

SPLITS = ("train", "val", "test")


def _deviation(counts: dict[str, int], total: int, fractions: dict[str, float]) -> float:
    """Largest gap, in percentage points, between a split's photo share and its target."""
    return max(abs(100 * counts[s] / total - 100 * fractions[s]) for s in SPLITS)


def rank_key(assignment: dict[str, str], stores: dict[str, dict], deviation: float) -> tuple:
    """Sort key (smaller is better) for choosing among assignments within the tolerance:
    1. the larger the smaller of the val and test store counts (test is the final measure, so it
       must not be left with few stores),
    2. then the more stores in val + test,
    3. then the more visits in val + test,
    4. then the smaller deviation from the target shares."""
    n_val = sum(1 for sp in assignment.values() if sp == "val")
    n_test = sum(1 for sp in assignment.values() if sp == "test")
    visits = sum(stores[s]["visits"] for s, sp in assignment.items() if sp != "train")
    return (-min(n_val, n_test), -(n_val + n_test), -visits, deviation)


def search_assignment(stores: dict[str, dict], fractions: dict[str, float], tolerance_points: float,
                      n_shuffles: int, seed: int, force_train: tuple[str, ...] = ()) -> dict:
    """Assign whole stores to splits.

    `stores` maps store code -> {"photos": n, "visits": n}. Stores are never split: all
    photos of a store land in one split, so two photos of the same shelf can never be in
    train and test. For each random store order, stores fill val and test while those are
    below their target (and stay within the tolerance), and the rest go to train. Among the
    assignments within the tolerance, `rank_key` picks the winner: the most balanced store count
    between val and test first, then the most stores, then the most visits, and only as the last
    tie-break the smallest deviation.
    """
    total = sum(s["photos"] for s in stores.values())
    targets = {s: fractions[s] * total for s in SPLITS}
    tol = tolerance_points / 100 * total
    rng = random.Random(seed)
    others = sorted(set(stores) - set(force_train))
    seen, best, best_key = set(), None, None
    for _ in range(n_shuffles):
        order = others[:]
        rng.shuffle(order)
        assign = {s: "train" for s in force_train}
        counts = {s: 0 for s in SPLITS}
        counts["train"] = sum(stores[s]["photos"] for s in force_train)
        for store in order:
            n = stores[store]["photos"]
            fits = [sp for sp in ("val", "test")
                    if targets[sp] - counts[sp] > 0 and counts[sp] + n <= targets[sp] + tol]
            sp = max(fits, key=lambda x: targets[x] - counts[x]) if fits else "train"
            assign[store] = sp
            counts[sp] += n
        dev = _deviation(counts, total, fractions)
        if dev > tolerance_points:
            continue
        signature = tuple(sorted(assign.items()))
        if signature in seen:
            continue
        seen.add(signature)
        key = rank_key(assign, stores, dev)
        if best_key is None or key < best_key:
            best, best_key = {"assignment": assign, "counts": counts, "deviation": dev}, key
    if best is None:
        raise ValueError("no assignment within the tolerance; raise n_shuffles or tolerance_points")
    best["n_valid_assignments_found"] = len(seen)
    return best


def validate_splits(splits: dict[str, list[str]], excluded: list[str], photo_store: dict[str, str],
                    id_regex: str, excluded_id_regex: str, no_store_code: str,
                    fractions: dict[str, float], tolerance_points: float) -> list[str]:
    """Return a list of error messages (empty = all good). `photo_store` maps every photo id
    in the mapping to its store code."""
    errors = []
    id_re, ex_re = re.compile(id_regex), re.compile(excluded_id_regex)
    for name, ids in splits.items():
        for i in ids:
            if not id_re.fullmatch(i):
                errors.append(f"{name}: malformed id {i!r}")
        if len(ids) != len(set(ids)):
            errors.append(f"{name}: duplicate ids in the file")
    for i in excluded:
        if not ex_re.fullmatch(i):
            errors.append(f"excluded: malformed id {i!r}")
    if len(excluded) != len(set(excluded)):
        errors.append("excluded: duplicate ids in the file")

    owner = {}
    for name, ids in splits.items():
        for i in ids:
            if i in owner and owner[i] != name:
                errors.append(f"photo {i} is in both {owner[i]} and {name}")
            owner[i] = name
            if i.startswith(no_store_code + "_"):
                errors.append(f"{name}: screenshot id {i} must never be in a split")
            if i not in photo_store:
                errors.append(f"{name}: id {i} is not in the mapping")
            if i in set(excluded):
                errors.append(f"{name}: id {i} is also excluded")
    for i in excluded:
        if i not in photo_store:
            errors.append(f"excluded: id {i} is not in the mapping")

    store_split = {}
    for i, name in owner.items():
        store = photo_store.get(i)
        if store is None:
            continue
        if store_split.setdefault(store, name) != name:
            errors.append(f"store {store} appears in both {store_split[store]} and {name}")

    expected = {i for i in photo_store if i not in set(excluded)}
    missing = expected - set(owner)
    if missing:
        errors.append(f"{len(missing)} non-excluded photos are in no split")
    total = len(owner)
    if total:
        counts = {s: len(splits.get(s, [])) for s in SPLITS}
        dev = _deviation(counts, total, fractions)
        if dev > tolerance_points:
            errors.append(f"split shares deviate {dev:.1f} points from the targets (tolerance {tolerance_points})")
    return errors
