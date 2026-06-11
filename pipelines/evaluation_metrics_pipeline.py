import pandas as pd

def print_rules(
    association_results,
    target_consequents: set[str] | None = None,
    min_antecedent_size: int = 2,
    max_antecedent_size: int = 3,
    max_rules: int = 20,
) -> None:
    """Print association rules in readable format."""

    # -------------------------------------------------------------------------
    # ETAPA 7 — Evaluation Metrics
    # -------------------------------------------------------------------------
    # Nesta etapa, as regras geradas pelo Apriori são avaliadas por meio das
    # principais métricas de regras de associação:
    #
    # Support:
    #     indica a frequência da ocorrência conjunta dos itens na base.
    #
    # Confidence:
    #     indica a probabilidade do consequente ocorrer dado o antecedente.
    #
    # Lift:
    #     indica se a associação é mais forte do que seria esperado ao acaso.
    #
    # Neste código, as métricas são extraídas e impressas para cada regra.
    # -------------------------------------------------------------------------

    extracted_rules = []

    for item in association_results:
        support = item.support

        for rule in item.ordered_statistics:
            antecedent = list(rule.items_base)
            consequent = list(rule.items_add)

            if len(antecedent) == 0 or len(consequent) == 0:
                continue

            if len(consequent) != 1:
                continue

            if len(antecedent) < min_antecedent_size:
                continue

            if len(antecedent) > max_antecedent_size:
                continue

            if target_consequents is not None:
                if consequent[0] not in target_consequents:
                    continue

            extracted_rules.append(
                {
                    "antecedent": antecedent,
                    "consequent": consequent,
                    "support": support,
                    "confidence": rule.confidence,
                    "lift": rule.lift,
                }
            )

    extracted_rules = sorted(
        extracted_rules,
        key=lambda x: (x["lift"], x["confidence"], x["support"]),
        reverse=True,
    )

    print(f"[PRINTING_RULES] Filtered rules: {len(extracted_rules)}\n")

    for rule in extracted_rules[:max_rules]:
        print("Rule:")
        print(f"  {rule['antecedent']} -> {rule['consequent']}")
        print(f"  Support: {rule['support']:.4f}")
        print(f"  Confidence: {rule['confidence']:.4f}")
        print(f"  Lift: {rule['lift']:.4f}")
        print("-" * 10)

def extract_apriori_rules_to_dataframe(
    association_results,
    target_consequents: set[str] | None = None,
    min_antecedent_size: int = 2,
    max_antecedent_size: int = 3,
) -> pd.DataFrame:
    """Extract filtered association rules into a pandas DataFrame."""

    extracted_rules = []

    for item in association_results:
        support = item.support

        for rule in item.ordered_statistics:
            antecedent = list(rule.items_base)
            consequent = list(rule.items_add)

            # Remove regras sem antecedente ou consequente
            if len(antecedent) == 0 or len(consequent) == 0:
                continue

            # Mantém apenas regras com um único consequente
            if len(consequent) != 1:
                continue

            # Controla o tamanho do antecedente
            if len(antecedent) < min_antecedent_size:
                continue

            if len(antecedent) > max_antecedent_size:
                continue

            # Filtra consequentes de interesse, se informado
            if target_consequents is not None:
                if consequent[0] not in target_consequents:
                    continue

            extracted_rules.append(
                {
                    "antecedent": ", ".join(antecedent),
                    "consequent": consequent[0],
                    "support": support,
                    "confidence": rule.confidence,
                    "lift": rule.lift,
                    "antecedent_size": len(antecedent),
                }
            )

    rules_df = pd.DataFrame(extracted_rules)

    if rules_df.empty:
        return rules_df

    rules_df = rules_df.sort_values(
        by=["lift", "confidence", "support"],
        ascending=[False, False, False],
    )

    return rules_df

def extract_fpgrowth_rules_to_dataframe(
    rules: pd.DataFrame,
    target_consequents: set[str] | None = None,
    min_antecedent_size: int = 2,
    max_antecedent_size: int = 3,
) -> pd.DataFrame:
    """Extract filtered FP-Growth association rules into a pandas DataFrame."""

    if rules.empty:
        return pd.DataFrame()

    extracted_rules = []

    for _, row in rules.iterrows():
        antecedent = list(row["antecedents"])
        consequent = list(row["consequents"])

        if len(antecedent) == 0 or len(consequent) == 0:
            continue

        if len(consequent) != 1:
            continue

        if len(antecedent) < min_antecedent_size:
            continue

        if len(antecedent) > max_antecedent_size:
            continue

        if target_consequents is not None:
            if consequent[0] not in target_consequents:
                continue

        extracted_rules.append(
            {
                "antecedent": ", ".join(sorted(antecedent)),
                "consequent": consequent[0],
                "support": row["support"],
                "confidence": row["confidence"],
                "lift": row["lift"],
                "antecedent_size": len(antecedent),
            }
        )

    rules_df = pd.DataFrame(extracted_rules)

    if rules_df.empty:
        return rules_df

    rules_df = rules_df.sort_values(
        by=["lift", "confidence", "support"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    return rules_df

def extract_eclat_rules_to_dataframe(
    association_results: list[dict],
    target_consequents: set[str] | None = None,
    min_antecedent_size: int = 2,
    max_antecedent_size: int = 3,
) -> pd.DataFrame:
    """Extract filtered ECLAT association rules into a pandas DataFrame."""

    extracted_rules = []

    for rule in association_results:
        antecedent = rule.get("antecedent", [])
        consequent = rule.get("consequent", [])

        if len(antecedent) == 0 or len(consequent) == 0:
            continue

        if len(consequent) != 1:
            continue

        if len(antecedent) < min_antecedent_size:
            continue

        if len(antecedent) > max_antecedent_size:
            continue

        if target_consequents is not None:
            if consequent[0] not in target_consequents:
                continue

        extracted_rules.append(
            {
                "antecedent": ", ".join(sorted(antecedent)),
                "consequent": consequent[0],
                "support": rule["support"],
                "confidence": rule["confidence"],
                "lift": rule["lift"],
                "antecedent_size": len(antecedent),
            }
        )

    rules_df = pd.DataFrame(extracted_rules)

    if rules_df.empty:
        return rules_df

    rules_df = rules_df.sort_values(
        by=["lift", "confidence", "support"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    return rules_df

def export_rules(rules_df: pd.DataFrame, output_path: str) -> None:
    """Export association rules to CSV or Excel."""

    if rules_df.empty:
        print("[EXPORT_RULES] No rules to export.")
        return

    if output_path.endswith(".csv"):
        rules_df.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"[EXPORT_RULES] Rules exported to CSV: {output_path}")

    elif output_path.endswith(".xlsx"):
        rules_df.to_excel(output_path, index=False)
        print(f"[EXPORT_RULES] Rules exported to Excel: {output_path}")

    else:
        raise ValueError("Unsupported file format. Use .csv or .xlsx")
    

