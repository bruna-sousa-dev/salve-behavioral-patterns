from apyori import apriori


class AprioriResultList(list):
    """Custom list to store Apriori metadata."""
    pass


def apriori_pipeline(records: list[list[str]]) -> list:
    """Apply Apriori algorithm."""

    min_support = 0.01
    min_confidence = 0.5
    min_lift = 1.5
    min_length = 2

    print("[APRIORI] Aplicando algoritmo Apriori...")
    print(
        f"[APRIORI] Parâmetros: "
        f"min_support={min_support}, "
        f"min_confidence={min_confidence}, "
        f"min_lift={min_lift}, "
        f"min_length={min_length}"
    )

    association_results = AprioriResultList(
        apriori(
            records,
            min_support=min_support,
            min_confidence=min_confidence,
            min_lift=min_lift,
            min_length=min_length,
        )
    )

    frequent_itemsets_count = len(association_results)

    rules_count = 0

    for item in association_results:
        for rule in item.ordered_statistics:
            antecedent = list(rule.items_base)
            consequent = list(rule.items_add)

            if len(antecedent) == 0 or len(consequent) == 0:
                continue

            rules_count += 1

    association_results.frequent_itemsets_count = frequent_itemsets_count
    association_results.rules_generated_count = rules_count
    association_results.min_support = min_support
    association_results.min_confidence = min_confidence
    association_results.min_lift = min_lift
    association_results.min_length = min_length

    print(f"[APRIORI] Itemsets frequentes encontrados: {frequent_itemsets_count}")
    print(f"[APRIORI] Regras geradas após filtros: {rules_count}")

    return association_results
