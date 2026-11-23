from defectinspect.data.ingest import build_manifest, load_split, write_manifest


def test_quality_checks_catch_planted_faults(synthetic_dir):
    vid, classes, records, report = build_manifest(synthetic_dir, 32, 0.15, 0.15, 42)
    assert len(classes) == 6
    assert report["rejected"] == {"corrupt": 1, "too_small": 1, "duplicate": 1}
    assert report["valid"] == 6 * 12


def test_split_is_deterministic_and_leak_free(synthetic_dir):
    a = build_manifest(synthetic_dir, 32, 0.15, 0.15, 42)
    b = build_manifest(synthetic_dir, 32, 0.15, 0.15, 42)
    assert a[0] == b[0]  # same data + seed -> same version id
    splits = {}
    for r in a[2]:
        if r["valid"]:
            assert splits.setdefault(r["sha256"], r["split"]) == r["split"]
    # Every class is present in every split.
    for c in a[1]:
        assert set(a[3]["per_class"][c]) == {"train", "val", "test"}


def test_seed_changes_version(synthetic_dir):
    a = build_manifest(synthetic_dir, 32, 0.15, 0.15, 42)
    b = build_manifest(synthetic_dir, 32, 0.15, 0.15, 7)
    assert a[0] != b[0]


def test_manifest_roundtrip(synthetic_dir, tmp_path):
    db = f"sqlite:///{tmp_path}/m.db"
    vid, classes, records, report = build_manifest(synthetic_dir, 32, 0.15, 0.15, 42)
    write_manifest(db, synthetic_dir, vid, classes, records, report)
    write_manifest(db, synthetic_dir, vid, classes, records, report)  # idempotent
    n = sum(len(load_split(db, vid, s)) for s in ("train", "val", "test"))
    assert n == report["valid"]
