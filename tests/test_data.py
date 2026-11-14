import pandas as pd

from src.data import (LABELS, build_dataset, clean_text, count_real_words,  # NEW
                      drop_near_duplicates, drop_uninformative, make_splits)  # NEW

TEMPLATE = ("I am writing to dispute the account listed below under the fair credit "
            "reporting act and demand an investigation of the account ") * 2


def _row(i, product, text):
    return {"Complaint ID": str(i), "Date received": "2024-09-01",
            "Product": product, "text": text}


def _raw():
    rows = []
    for i in range(1, 4):                                   # 3 unique mortgage
        rows.append(_row(i, "Mortgage", f"Unique mortgage complaint number {i} " + "word " * 40))
    rows.append(_row(4, "Mortgage", "Exact duplicate text about a mortgage payment problem"))
    rows.append(_row(5, "Mortgage", "Exact duplicate text about a mortgage payment problem"))
    rows.append(_row(6, "Mortgage", "Same text filed under two different products"))
    rows.append(_row(7, "Student loan", "Same text filed under two different products"))
    rows.append(_row(8, "Debt collection", TEMPLATE + "account one"))
    rows.append(_row(9, "Debt collection", TEMPLATE + "account two"))
    rows.append(_row(10, "Prepaid card", "Unselected product complaint " + "word " * 40))
    for i in range(11, 16):                                  # 5 unique credit reporting
        rows.append(_row(i, "Credit reporting or other personal consumer reports",
                         f"Credit reporting complaint number {i} " + "word " * 40))
    return pd.DataFrame(rows)


def test_clean_text_replaces_dates_and_redactions():
    out = clean_text("On XX/XX/XXXX XXXX XXXX called me")
    assert out == "On [DATE] [REDACTED] called me"
    assert clean_text("Paid {$1,200.00} on XX/XX/2020") == "Paid {$1,200.00} on [DATE]"


def test_clean_text_collapses_whitespace_and_keeps_amounts():
    assert clean_text("  too   many\n spaces ") == "too many spaces"
    assert "{$35.00}" in clean_text("They charged {$35.00} twice")


def test_clean_text_is_idempotent():
    once = clean_text("XXXX called on XX/XX/XXXX about {$20.00}")
    assert clean_text(once) == once


def test_build_dataset_removes_conflicts_duplicates_and_applies_cap():
    out, report = build_dataset(_raw(), cap=4)
    counts = out["label"].value_counts().to_dict()
    # Mortgage: 3 unique + 1 survivor of the exact-duplicate pair = 4 (the conflict pair is gone).
    # Debt collection: 1 survivor of the template pair.
    # Credit reporting: 5 unique complaints, capped at 4.
    # Student loan: its only row was a label conflict. Prepaid card is not a selected product.
    assert counts == {"Mortgage": 4, "Debt collection": 1, "Credit reporting": 4}
    assert report["raw_rows"] == 15
    assert report["after_label_conflicts"] == 13
    assert report["after_template_dedup"] == 11
    assert report["after_product_filter"] == 10
    assert report["final_rows"] == len(out) == 9


def test_build_dataset_is_deterministic():
    a, _ = build_dataset(_raw(), cap=3)
    b, _ = build_dataset(_raw(), cap=3)
    pd.testing.assert_frame_equal(a, b)


def test_build_dataset_labels_are_valid():
    out, _ = build_dataset(_raw(), cap=3)
    assert set(out["label"]) <= set(LABELS)
    assert out["label_id"].between(0, len(LABELS) - 1).all()
    assert out["complaint_id"].is_unique


# ---- near-duplicate removal -------------------------------------------------------  # NEW
BASE = ("The credit bureau keeps reporting an account that is not mine and refuses to delete "
        "the inaccurate tradeline from my report despite three disputes and a notarized "
        "identity theft affidavit sent by certified mail months ago")


def _labelled(rows):
    return pd.DataFrame([{"complaint_id": str(i), "label": lab, "label_id": 0, "text": t}
                         for i, (lab, t) in enumerate(rows)])


def _near_dup_frame():
    return _labelled([
        ("A", BASE + " to Experian"),
        ("A", BASE + " to Equifax"),                      # near-copy of the first A complaint
        ("A", "My mortgage servicer misapplied two escrow payments and now claims I am "
              "delinquent, so I need help getting the payment history corrected soon"),
        ("A", "A debt collector keeps phoning my workplace about a loan I already paid "
              "off years ago and will not send written validation of the amount"),
        ("B", BASE + " to Experian"),                     # same text in ANOTHER class: kept
    ])


def test_drop_near_duplicates_keeps_one_of_each_near_copy_pair():  # NEW
    out = drop_near_duplicates(_near_dup_frame(), min_df=1, max_df=1.0)
    a = out[out["label"] == "A"]
    assert len(a) == 3                                    # 4 -> 3: one near-copy removed
    assert a["text"].str.startswith("The credit bureau").sum() == 1
    assert (out["label"] == "B").sum() == 1               # classes are never compared


def test_drop_near_duplicates_is_deterministic():  # NEW
    a = drop_near_duplicates(_near_dup_frame(), min_df=1, max_df=1.0)
    b = drop_near_duplicates(_near_dup_frame(), min_df=1, max_df=1.0)
    pd.testing.assert_frame_equal(a, b)


def test_drop_near_duplicates_keeps_everything_at_threshold_one():  # NEW
    out = drop_near_duplicates(_near_dup_frame(), threshold=1.01, min_df=1, max_df=1.0)
    assert len(out) == 5


def test_count_real_words_ignores_placeholders():  # NEW
    s = pd.Series(["[REDACTED] [DATE] [REDACTED]", "I did not authorize [REDACTED] [DATE]",
                   "There are unauthorized inquiries on my credit report."])
    assert count_real_words(s).tolist() == [0, 3, 8]


def test_drop_uninformative_removes_short_and_placeholder_only_rows():  # NEW
    df = pd.DataFrame({"text": ["[REDACTED] [DATE] [REDACTED]", "Dignify [REDACTED]",
                                "There are unauthorized inquiries on my credit report."]})
    out = drop_uninformative(df)
    assert out["text"].tolist() == ["There are unauthorized inquiries on my credit report."]


# ---- splits ----------------------------------------------------------------------
def _toy():
    n0, n1 = 200, 100
    return pd.DataFrame({"complaint_id": [str(i) for i in range(n0 + n1)],
                         "label_id": [0] * n0 + [1] * n1, "text": "t"})


def test_make_splits_no_overlap_and_full_coverage():
    train, val, test = make_splits(_toy())
    assert len(train) + len(val) + len(test) == 300
    ids = [set(s["complaint_id"]) for s in (train, val, test)]
    assert not (ids[0] & ids[1]) and not (ids[0] & ids[2]) and not (ids[1] & ids[2])
    assert abs(len(val) - 45) <= 1 and abs(len(test) - 45) <= 1


def test_make_splits_is_stratified():
    for split in make_splits(_toy()):
        assert abs(split["label_id"].mean() - 1 / 3) < 0.02


def test_make_splits_is_reproducible():
    a = make_splits(_toy())
    b = make_splits(_toy())
    for x, y in zip(a, b):
        pd.testing.assert_frame_equal(x, y)