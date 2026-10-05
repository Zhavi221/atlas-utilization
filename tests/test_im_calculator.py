import awkward as ak

from services.calculations.im_calculator import IMCalculator


def test_light_jets_are_not_limited_by_max_combination_count():
    events = ak.Array({
        "Muons": [[1, 2]],
        "Jets": [[1, 2, 3, 4, 5]],
        "BJets": [[1]],
    })
    calculator = IMCalculator(
        events,
        min_events_per_fs=1,
        min_k=1,
        max_k=4,
        min_n=2,
        max_n=4,
    )

    final_state = "2m_5j_1b"
    assert calculator.final_state_counts()[final_state] == 1
    assert list(calculator.group_by_final_state()) == [final_state]
    assert len(calculator.get_events_for_final_state(final_state)) == 1


def test_multi_digit_light_jet_count_is_used_in_combination_check():
    calculator = IMCalculator(
        ak.Array([]),
        min_events_per_fs=1,
        min_k=1,
        max_k=4,
        min_n=1,
        max_n=4,
    )

    assert calculator.does_final_state_contain_combination(
        "2m_10j_1b", {"Jets": 10}
    )
    assert not calculator.does_final_state_contain_combination(
        "2m_10j_1b", {"Jets": 11}
    )
