from itertools import combinations

class EclatResultList(list):
    """Custom list to store ECLAT metadata."""
    pass

def eclat_pipeline(records: list[list[str]]) -> list[dict]:
    """Apply ECLAT algorithm."""

    # -------------------------------------------------------------------------
    # ETAPA 6 — Application of algorithms: ECLAT
    # -------------------------------------------------------------------------
    # Nesta etapa, o algoritmo ECLAT é aplicado sobre a base transacional.
    #
    # O ECLAT busca conjuntos frequentes de itens utilizando uma representação
    # vertical da base de dados.
    #
    # Em vez de percorrer repetidamente todas as transações, cada item é
    # associado ao conjunto de IDs das transações em que ocorre. Os suportes são
    # calculados por meio da interseção entre esses conjuntos.
    #
    # Parâmetros principais:
    #
    # min_support:
    #     frequência mínima para que um itemset seja considerado frequente.
    #
    # min_confidence:
    #     força condicional mínima da regra.
    #
    # min_lift:
    #     grau mínimo de associação acima do esperado ao acaso.
    #
    # min_length:
    #     tamanho mínimo dos itemsets considerados.
    # -------------------------------------------------------------------------

    min_support = 0.01
    min_confidence = 0.5
    min_lift = 1.5
    min_length = 2

    print("[ECLAT] Aplicando algoritmo ECLAT...")
    print(
        f"[ECLAT] Parâmetros: "
        f"min_support={min_support}, "
        f"min_confidence={min_confidence}, "
        f"min_lift={min_lift}, "
        f"min_length={min_length}"
    )

    total_transactions = len(records)
    min_support_count = max(1, int(min_support * total_transactions))

    # -------------------------------------------------------------------------
    # Construção da representação vertical:
    # item -> conjunto de IDs das transações onde o item aparece
    # -------------------------------------------------------------------------
    vertical_db: dict[str, set[int]] = {}

    for transaction_id, transaction in enumerate(records):
        for item in set(transaction):
            if item not in vertical_db:
                vertical_db[item] = set()
            vertical_db[item].add(transaction_id)

    # -------------------------------------------------------------------------
    # Filtra itens frequentes de tamanho 1
    # -------------------------------------------------------------------------
    frequent_items = {
        frozenset([item]): transaction_ids
        for item, transaction_ids in vertical_db.items()
        if len(transaction_ids) >= min_support_count
    }

    frequent_itemsets: dict[frozenset[str], set[int]] = {}

    # -------------------------------------------------------------------------
    # Função recursiva ECLAT
    # -------------------------------------------------------------------------
    def eclat_recursive(
        prefix: frozenset[str],
        items: list[tuple[frozenset[str], set[int]]]
    ) -> None:
        for i, (itemset_i, tidset_i) in enumerate(items):
            new_itemset = prefix | itemset_i
            new_tidset = tidset_i

            if len(new_tidset) >= min_support_count:
                frequent_itemsets[new_itemset] = new_tidset

                suffix = []
                for itemset_j, tidset_j in items[i + 1:]:
                    combined_itemset = itemset_i | itemset_j
                    combined_tidset = tidset_i & tidset_j

                    if len(combined_tidset) >= min_support_count:
                        suffix.append((combined_itemset, combined_tidset))

                if suffix:
                    eclat_recursive(new_itemset, suffix)

    sorted_frequent_items = sorted(
        frequent_items.items(),
        key=lambda x: len(x[1]),
        reverse=True
    )

    eclat_recursive(frozenset(), sorted_frequent_items)

    # -------------------------------------------------------------------------
    # Remove itemsets menores que min_length
    # -------------------------------------------------------------------------
    frequent_itemsets = {
        itemset: tidset
        for itemset, tidset in frequent_itemsets.items()
        if len(itemset) >= min_length
    }

    # -------------------------------------------------------------------------
    # Geração das regras de associação
    # -------------------------------------------------------------------------
    rules = []

    support_cache = {
        itemset: len(tidset) / total_transactions
        for itemset, tidset in frequent_itemsets.items()
    }

    # Inclui suporte dos itemsets unitários no cache
    for itemset, tidset in frequent_items.items():
        support_cache[itemset] = len(tidset) / total_transactions

    for itemset in frequent_itemsets:
        if len(itemset) < 2:
            continue

        items = list(itemset)

        for r in range(1, len(items)):
            for antecedent_tuple in combinations(items, r):
                antecedent = frozenset(antecedent_tuple)
                consequent = itemset - antecedent

                if not consequent:
                    continue

                support_itemset = support_cache.get(itemset)
                support_antecedent = support_cache.get(antecedent)
                support_consequent = support_cache.get(consequent)

                if support_itemset is None:
                    continue

                if support_antecedent is None:
                    support_antecedent = compute_support(
                        antecedent,
                        records,
                        total_transactions
                    )
                    support_cache[antecedent] = support_antecedent

                if support_consequent is None:
                    support_consequent = compute_support(
                        consequent,
                        records,
                        total_transactions
                    )
                    support_cache[consequent] = support_consequent

                confidence = support_itemset / support_antecedent

                if support_consequent == 0:
                    continue

                lift = confidence / support_consequent

                if confidence >= min_confidence and lift >= min_lift:
                    rules.append({
                        "items": sorted(itemset),
                        "antecedent": sorted(antecedent),
                        "consequent": sorted(consequent),
                        "support": support_itemset,
                        "confidence": confidence,
                        "lift": lift,
                    })

    rules = sorted(
        rules,
        key=lambda rule: (
            rule["lift"],
            rule["confidence"],
            rule["support"]
        ),
        reverse=True
    )

    result = EclatResultList(rules)

    result.frequent_itemsets_count = len(frequent_itemsets)
    result.rules_generated_count = len(rules)
    result.min_support = min_support
    result.min_confidence = min_confidence
    result.min_lift = min_lift
    result.min_length = min_length

    print(f"[ECLAT] Itemsets frequentes encontrados: {len(frequent_itemsets)}")
    print(f"[ECLAT] Regras geradas após filtros: {len(rules)}")

    return result


def compute_support(
    itemset: frozenset[str],
    records: list[list[str]],
    total_transactions: int
) -> float:
    """Compute support for an itemset."""

    count = 0

    for transaction in records:
        if itemset.issubset(set(transaction)):
            count += 1

    return count / total_transactions
