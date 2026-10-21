from src.data.splits import rank_key, search_assignment, validate_splits

FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}
ID_RE = r"^s\d{2}_[0-9a-f]{8}$"
EX_RE = r"^s(\d{2}|x)_[0-9a-f]{8}$"


def _stores():
    sizes = [48, 34, 25, 21, 18, 17, 13, 12, 11, 10, 9, 9, 7, 6, 4, 3, 2, 2, 2]
    return {f"s{i + 1:02d}": {"photos": n, "visits": 1 + i % 3} for i, n in enumerate(sizes)}


def test_search_is_deterministic_and_within_tolerance():
    stores = _stores()
    a = search_assignment(stores, FRACTIONS, 5, 500, seed=1, force_train=("s01",))
    b = search_assignment(stores, FRACTIONS, 5, 500, seed=1, force_train=("s01",))
    assert a["assignment"] == b["assignment"]
    assert a["deviation"] <= 5
    assert a["assignment"]["s01"] == "train"
    assert set(a["assignment"].values()) == {"train", "val", "test"}


def test_every_store_is_in_exactly_one_split():
    stores = _stores()
    result = search_assignment(stores, FRACTIONS, 5, 300, seed=3, force_train=("s01",))
    assert set(result["assignment"]) == set(stores)
    assert sum(result["counts"].values()) == sum(s["photos"] for s in stores.values())


def _ids(store, n, start=0):
    return [f"{store}_{start + k:08x}" for k in range(n)]


def _valid_case():
    splits = {"train": _ids("s01", 7), "val": _ids("s02", 2), "test": _ids("s03", 1)}
    photo_store = {i: i.split("_")[0] for ids in splits.values() for i in ids}
    excluded = ["sx_0000000a"]
    photo_store["sx_0000000a"] = "sx"
    return splits, excluded, photo_store


def _check(splits, excluded, photo_store, tol=20):
    return validate_splits(splits, excluded, photo_store, ID_RE, EX_RE, "sx", FRACTIONS, tol)


def test_valid_splits_pass():
    assert _check(*_valid_case()) == []


def test_store_in_two_splits_is_an_error():
    splits, excluded, photo_store = _valid_case()
    extra = "s01_ffffffff"
    splits["val"].append(extra)
    photo_store[extra] = "s01"
    assert any("store s01 appears in both" in e for e in _check(splits, excluded, photo_store))


def test_photo_in_two_splits_is_an_error():
    splits, excluded, photo_store = _valid_case()
    splits["test"].append(splits["train"][0])
    assert any("is in both" in e for e in _check(splits, excluded, photo_store))


def test_screenshot_id_in_a_split_is_an_error_and_fails_the_regex():
    splits, excluded, photo_store = _valid_case()
    splits["val"].append("sx_0000000a")
    errors = _check(splits, excluded, photo_store)
    assert any("malformed" in e for e in errors) and any("screenshot" in e for e in errors)


def test_sx_ids_are_allowed_in_excluded_only():
    splits, excluded, photo_store = _valid_case()
    assert _check(splits, excluded, photo_store) == []
    assert any("malformed" in e for e in _check(splits, ["x_123"], photo_store))


def test_unassigned_photo_and_unknown_id_are_errors():
    splits, excluded, photo_store = _valid_case()
    photo_store["s04_00000001"] = "s04"      # in the mapping but in no split
    splits["train"].append("s09_deadbeef")  # in a split but not in the mapping
    errors = _check(splits, excluded, photo_store)
    assert any("in no split" in e for e in errors) and any("not in the mapping" in e for e in errors)


def test_balance_rule_prefers_the_smaller_of_val_and_test_store_counts():
    stores = {f"s{i:02d}": {"photos": 1, "visits": 1} for i in range(1, 12)}

    def assign(n_val, n_test):
        a = {s: "train" for s in stores}
        names = sorted(stores)
        for s in names[:n_val]:
            a[s] = "val"
        for s in names[n_val:n_val + n_test]:
            a[s] = "test"
        return a

    unbalanced = rank_key(assign(6, 1), stores, 0.0)  # 7 stores held out, but test has only 1
    balanced = rank_key(assign(3, 3), stores, 0.0)    # 6 stores held out, min is 3
    assert balanced < unbalanced                      # balance beats the larger total
    # equal minimum: more stores in total wins
    assert rank_key(assign(3, 4), stores, 5.0) < rank_key(assign(3, 3), stores, 0.0)
    # equal stores: more visits wins, then the smaller deviation
    more_visits = dict(stores, s01={"photos": 1, "visits": 5})
    assert rank_key(assign(3, 3), more_visits, 4.0) < rank_key(assign(3, 3), stores, 0.0)
    assert rank_key(assign(3, 3), stores, 0.5) < rank_key(assign(3, 3), stores, 1.0)


def test_search_returns_the_most_balanced_valid_assignment():
    stores = _stores()
    best = search_assignment(stores, FRACTIONS, 5, 2000, seed=7, force_train=("s01",))
    n_val = sum(1 for sp in best["assignment"].values() if sp == "val")
    n_test = sum(1 for sp in best["assignment"].values() if sp == "test")
    # with these store sizes a 3-vs-1 split would be allowed; the rule must not pick one that is lopsided
    assert min(n_val, n_test) >= max(n_val, n_test) - 3
