# SPDX-License-Identifier: Apache-2.0
"""Regression tests for opt-in integration-test collection."""

from tests import conftest


class _Config:
    def __init__(self, *, server_url=None, run_slow=False):
        self.options = {"--server-url": server_url, "--run-slow": run_slow}

    def getoption(self, name):
        return self.options[name]


class _Item:
    def __init__(self, *keywords):
        self.keywords = dict.fromkeys(keywords, True)
        self.markers = []

    def add_marker(self, marker):
        self.markers.append(marker)

    def get_closest_marker(self, name):
        return name if name in self.keywords else None


def test_explicit_server_url_enables_all_integration_tests():
    items = [_Item("integration"), _Item("integration", "slow")]

    conftest.pytest_collection_modifyitems(
        _Config(server_url="http://localhost:8000"), items
    )

    assert [item.markers for item in items] == [[], []]


def test_integration_tests_remain_skipped_without_server_url():
    item = _Item("integration")

    conftest.pytest_collection_modifyitems(_Config(), [item])

    assert len(item.markers) == 1
    assert item.markers[0].mark.name == "skip"
