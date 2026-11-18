from policy_extractor.sampling import UNKNOWN_TYPE, select_spot_checks


def test_every_policy_type_with_an_approved_record_gets_at_least_one_sample():
    approved = {f"A{i}": "auto" for i in range(20)} | {"H1": "home", "U1": "umbrella"}
    selected = select_spot_checks(approved, share=0.1, seed=0)
    assert {approved[doc] for doc in selected} == {"auto", "home", "umbrella"}
    assert sum(approved[doc] == "auto" for doc in selected) == 2


def test_sampling_is_deterministic_and_independent_of_input_order():
    approved = {f"D{i}": ("auto" if i % 2 else "home") for i in range(30)}
    first = select_spot_checks(approved, share=0.2, seed=7)
    reordered = dict(reversed(list(approved.items())))
    assert select_spot_checks(reordered, share=0.2, seed=7) == first
    assert select_spot_checks(approved, share=0.2, seed=8) != first


def test_unknown_type_is_its_own_stratum_and_empty_input_selects_nothing():
    assert select_spot_checks({"X": None}, share=0.1, seed=0) == {"X"}
    assert UNKNOWN_TYPE.startswith("<")
    assert select_spot_checks({}, share=0.1, seed=0) == set()


def test_a_share_of_zero_turns_sampling_off():
    assert select_spot_checks({"A": "auto", "H": "home"}, share=0.0, seed=0) == set()
