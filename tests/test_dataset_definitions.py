"""
Consistency checks for the hand-written channel definitions of every registered dataset.

All lookups during processing (channel_types, channel_groups, intra- and inter-dataset mapping)
are exact string matches, so a typo silently disables harmonization, filtering or rescaling for
that channel. These tests catch such mismatches without needing any data on disk.

Run with: python -m pytest tests
"""

import pytest

import datasets  # noqa: F401  (discovers all dataset modules)
from datasets.registry import DatasetRegistry, get_dataset


def _dataset_params():
    """One test case per dataset (and per subset for datasets like BDSP that prompt for one)."""
    params = []
    for name in DatasetRegistry.list_datasets():
        try:
            cls = get_dataset(name)
        except ImportError as e:
            params.append(pytest.param((name, None), id=name, marks=pytest.mark.skip(reason=f"cannot import: {e}")))
            continue
        for subset in getattr(cls, "SUBSETS", [None]):
            params.append(pytest.param((name, subset), id=f"{name}-{subset}" if subset else name))
    return params


@pytest.fixture(params=_dataset_params())
def dataset(request, monkeypatch):
    name, subset = request.param
    if subset is not None:
        monkeypatch.setattr("builtins.input", lambda *_: subset)
    return get_dataset(name)()


def _raw_names(dataset):
    """All original channel names known for a dataset: channel_names plus every intra-dataset alias."""
    names = set(dataset.channel_names)
    for aliases in dataset.intra_dataset_mapping.values():
        names.update(aliases)
    return names


def _intra_mapped(dataset, channel):
    """Name of a channel after the intra-dataset step of BaseDataset.map_channel."""
    for key, aliases in dataset.intra_dataset_mapping.items():
        if channel in aliases:
            return key
    return channel


def test_inter_mapping_keys_are_reachable(dataset):
    """Every inter-dataset key must be produced by some raw channel name, otherwise it never maps anything."""
    reachable = {_intra_mapped(dataset, ch) for ch in _raw_names(dataset)}
    unreachable = [key for key in dataset.inter_dataset_mapping if key not in reachable]
    assert not unreachable, (
        f"inter_dataset_mapping keys not matching any intra_dataset_mapping key or channel name: {unreachable}"
    )


def test_intra_aliases_are_unique(dataset):
    """An alias listed under two intra-dataset keys is silently mapped to the first one only."""
    owner = {}
    duplicates = {}
    for key, aliases in dataset.intra_dataset_mapping.items():
        for alias in aliases:
            if alias in owner and owner[alias] != key:
                duplicates[alias] = [owner[alias], key]
            owner.setdefault(alias, key)
    assert not duplicates, f"aliases listed under multiple intra_dataset_mapping keys: {duplicates}"


def test_channel_groups_use_raw_names(dataset):
    """channel_groups are looked up with the original channel name, so entries must be raw names."""
    known = _raw_names(dataset)
    unknown = sorted({ch for group in dataset.channel_groups.values() for ch in group} - known)
    assert not unknown, f"channel_groups entries that are no known channel name (never matched): {unknown}"


def test_channel_types_use_raw_names(dataset):
    """channel_types are looked up with the original channel name, so entries must be raw names."""
    known = _raw_names(dataset)
    unknown = sorted({ch for chs in dataset.channel_types.values() for ch in chs} - known)
    assert not unknown, f"channel_types entries that are no known channel name (never matched): {unknown}"


def test_every_channel_has_exactly_one_type(dataset):
    """SignalProcessor raises for channels without a type, and picks the first type if listed twice."""
    counts = {ch: sum(ch in chs for chs in dataset.channel_types.values()) for ch in dataset.channel_names}
    untyped = sorted(ch for ch, n in counts.items() if n == 0)
    multi_typed = sorted(ch for ch, n in counts.items() if n > 1)
    assert not untyped, f"channels missing in channel_types: {untyped}"
    assert not multi_typed, f"channels listed under multiple channel_types: {multi_typed}"


def test_every_channel_in_at_most_one_group(dataset):
    """Filtering and rescaling use the first matching group, so a channel in two groups is ambiguous."""
    multi_grouped = sorted(
        ch for ch in dataset.channel_names
        if sum(ch in group for group in dataset.channel_groups.values()) > 1
    )
    assert not multi_grouped, f"channels listed under multiple channel_groups: {multi_grouped}"
