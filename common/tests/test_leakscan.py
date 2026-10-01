from common import leakscan

CAN = "RLCANARY-x-0123456789abcdef"


def test_canary_anywhere_including_model_fields():
    r = leakscan.scan({"text": "blah rlcanary-X-0123456789ABCDEF"}, CAN, [], {"text"})
    assert r["leak"] and r["reasons"] == ["canary"]


def test_leak_string_outside_model_field_case_and_whitespace_insensitive():
    r = leakscan.scan({"label": "The answer is  sacra MENTO"}, CAN, ["Sacramento"], {"text"})
    assert r["leak"]


def test_leak_string_in_model_field_is_behavioral_exposure_not_leak():
    r = leakscan.scan({"text": ["Sacramento!", "sacramento"], "n": 2}, CAN, ["Sacramento"], {"text"})
    assert not r["leak"] and r["behavioral_exposure"] == 2


def test_nested_model_field():
    r = leakscan.scan({"results": [{"completion": "it is Sacramento", "prob": 0.5}]}, CAN, ["Sacramento"], {"completion"})
    assert not r["leak"] and r["behavioral_exposure"] == 1


def test_numbers_and_lists_rendered_as_text():
    assert leakscan.scan({"freqs": [14, 35, 41]}, CAN, ["14, 35, 41"])["leak"]
    assert leakscan.scan({"freqs": [14, 35, 41]}, CAN, ["[14,35,41]"])["leak"]
    # digit boundaries: "14,35" must not fire inside "114,356"
    assert not leakscan.scan({"freqs": [114, 356]}, CAN, ["14,35"])["leak"]


def test_agent_supplied_string_is_exempt_but_canary_is_not():
    resp = {"prompt": "is it Sacramento?", "logits": [1.0]}
    assert leakscan.scan(resp, CAN, ["Sacramento"], (), agent_text='{"prompt": "is it Sacramento?"}')["leak"] is False
    assert leakscan.scan(resp, CAN, ["Sacramento"], ())["leak"] is True
    assert leakscan.scan({"x": CAN}, CAN, [], (), agent_text=CAN)["leak"] is True


def test_short_leak_strings_rejected():
    assert leakscan.check_leak_strings(["ab", "fine-string"])
    assert not leakscan.check_leak_strings(["abc"])


def test_scan_text():
    t = "notes: answer Sacramento\n" + CAN.lower()
    r = leakscan.scan_text(t, CAN, ["sacramento", "nothing"])
    assert r["canary"] and r["leak_strings"] == [0]
