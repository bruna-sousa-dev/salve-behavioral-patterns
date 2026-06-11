from mlxtend.frequent_patterns import fpgrowth, association_rules
import pandas as pd


def fpgrowth_pipeline(records: list[list[str]]) -> pd.DataFrame:
    """Apply FP-Growth algorithm."""

    min_support = 0.01
    min_confidence = 0.5
    min_lift = 1.5
    min_length = 2

    print("[FP-GROWTH] Aplicando algoritmo FP-Growth...")
    print(
        f"[FP-GROWTH] Parâmetros: "
        f"min_support={min_support}, "
        f"min_confidence={min_confidence}, "
        f"min_lift={min_lift}, "
        f"min_length={min_length}"
    )

    all_items = sorted(set(item for transaction in records for item in transaction))

    encoded_records = []
    for transaction in records:
        transaction_set = set(transaction)
        encoded_records.append({
            item: item in transaction_set for item in all_items
        })

    df_encoded = pd.DataFrame(encoded_records)

    frequent_itemsets = fpgrowth(
        df_encoded,
        min_support=min_support,
        use_colnames=True
    )

    frequent_itemsets["length"] = frequent_itemsets["itemsets"].apply(len)

    rules = association_rules(
        frequent_itemsets,
        metric="confidence",
        min_threshold=min_confidence
    )

    rules["antecedent_len"] = rules["antecedents"].apply(len)
    rules["consequent_len"] = rules["consequents"].apply(len)
    rules["rule_len"] = rules["antecedent_len"] + rules["consequent_len"]

    rules = rules[
        (rules["lift"] >= min_lift) &
        (rules["rule_len"] >= min_length)
    ]

    rules = rules.sort_values(
        by=["lift", "confidence", "support"],
        ascending=False
    ).reset_index(drop=True)

    rules.attrs["frequent_itemsets_count"] = len(frequent_itemsets)
    rules.attrs["rules_generated_count"] = len(rules)
    rules.attrs["min_support"] = min_support
    rules.attrs["min_confidence"] = min_confidence
    rules.attrs["min_lift"] = min_lift
    rules.attrs["min_length"] = min_length

    print(f"[FP-GROWTH] Itemsets frequentes encontrados: {len(frequent_itemsets)}")
    print(f"[FP-GROWTH] Regras geradas após filtros: {len(rules)}")

    return rules