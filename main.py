type_dataset = "synthetic" # observational or synthetic

step = "visualizacao_distribuicao_tempo_resposta"
"""
Definir qual pipeline executar
Opções: 
>>> Análise exploratória de dados (EDA):
-> "visualizacao_distribuicao_tempo_resposta": 
      visualiza a distribuição do tempo de resposta usando um histograma com KDE.

-> "visualizacao_boxplot_tempo_resposta": 
      visualiza o boxplot do tempo de resposta para identificar outliers e dispersão.

-> "visualizacao_tempo_resposta_por_hora":
      visualiza a taxa de atendimento por hora para verificar variações ao longo do dia.

-> "visualizacao_taxa_atendimento_dia_util_vs_fim_de_semana":
      visualiza a taxa de atendimento em dias úteis vs fins de semana.

-> "visualizacao_matriz_correlacao":
      visualiza a matriz de correlação entre as variáveis numéricas.

-> "visualizacao_todas":
      Executa todas as visualizações de EDA em sequência.

-> "dataset_quality_summary":
      Exibe um resumo da qualidade dos dados, incluindo contagem de registros, taxa de atendimento, valores ausentes e duplicatas.

-> "response_time_statistics":
      Exibe estatísticas descritivas do tempo de resposta, como média, mediana, desvio padrão e quartis.

-> "boxplot_outlier_statistics":
      Exibe estatísticas relacionadas a outliers no tempo de resposta, como contagem de outliers e limites do boxplot.

-> "service_rate_by_hour":
      Exibe a taxa de atendimento por hora, mostrando a proporção de alertas atendidos em cada hora do dia.

-> "service_rate_weekday_vs_weekend":
      Exibe a taxa de atendimento em dias úteis vs fins de semana, mostrando a proporção de alertas atendidos em cada categoria.

-> "correlation_statistics":
      Exibe a matriz de correlação entre as variáveis numéricas, mostrando os coeficientes de correlação.

-> "run_all_statistics":
      Executa todas as análises estatísticas de EDA em sequência.
      
-> "pipeline_geracao_regras":
      Executa a pipeline de geração de regras, que inclui a aplicação do algoritmo Apriori para 
      gerar regras de associação a partir dos dados processados.
"""

from pipelines import (
    cleaning_preprocess_pipeline,
    synthetic_dataset_expansion_pipeline,
    feature_engineering_pipeline,
    discretization_categorical_pipeline,
    transactional_format_pipeline,
    apriori_pipeline,
    fpgrowth_pipeline,
    eclat_pipeline,
    print_rules,
    extract_apriori_rules_to_dataframe,
    extract_fpgrowth_rules_to_dataframe,
    extract_eclat_rules_to_dataframe,
    export_rules,
    run_with_timer,
    build_experiment_summary_row,
    export_experiment_summary,
)

target_consequents = {
    "response_time_class=not_served",
    "response_time_class=fast_le_60s",
    "response_time_class=normal_61_180s",
    "response_time_class=slow_gt_180s",
}
min_antecedent_size = 2
max_antecedent_size = 3
experiment_summary_rows = []

if type_dataset == "observational":

    print("-" * 100)
    print(f"[DATASET] Tipo do dataset: {type_dataset}")

    print("[CLEANING] Executando pipeline de limpeza...")
    cleaning_preprocess_pipeline()
    print("[CLEANING] OK.")
    print("-" * 100)

    print("[FEATURE_ENGINEERING] Executando pipeline de engenharia de features...")
    feature_engineering_pipeline(type_dataset="observational")
    print("[FEATURE_ENGINEERING] OK.")
    print("-" * 100)

    print("[DISCRETIZATION_CATEGORICAL] Executando pipeline de discretização categórica...")
    discretization_categorical_pipeline(type_dataset="observational")
    print("[DISCRETIZATION_CATEGORICAL] OK.")
    print("-" * 100)

    print("[TRANSACTIONAL_FORMAT] Executando pipeline para formato transacional...")
    records = transactional_format_pipeline(type_dataset="observational")
    print("[TRANSACTIONAL_FORMAT] OK.")
    print("-" * 100)

    print("[ARM] Executando pipelines ARM...")
    apriori_results, apriori_time = run_with_timer(
        "APRIORI",
        apriori_pipeline,
        records,
    )
    print("[ARM_APRIORI] OK.")

    fpgrowth_results, fpgrowth_time = run_with_timer(
        "FP-GROWTH",
        fpgrowth_pipeline,
        records,
    )
    print("[ARM_FPGROWTH] OK.")

    eclat_results, eclat_time = run_with_timer(
        "ECLAT",
        eclat_pipeline,
        records,
    )
    print("[ARM_ECLAT] OK.")
    print("-" * 100)

    # print("[PRINTING_RULES] Printando regras de associação...")
    # print("[PRINTING_RULES_APRIORI]:")
    # print_rules(
    #     apriori_results,
    #     target_consequents={
    #         "response_time_class=not_served",
    #         "response_time_class=fast_le_60s",
    #         "response_time_class=normal_61_180s",
    #         "response_time_class=slow_gt_180s",
    #     },
    #     min_antecedent_size=2,
    #     max_antecedent_size=3,
    #     max_rules=1,
    # )
    # print("[PRINTING_RULES_APRIORI] OK.")

    # print("[PRINTING_RULES_FPGROWTH]:")
    # print_rules(
    #     fpgrowth_results,
    #     target_consequents={
    #         "response_time_class=not_served",
    #         "response_time_class=fast_le_60s",
    #         "response_time_class=normal_61_180s",
    #         "response_time_class=slow_gt_180s",
    #     },
    #     min_antecedent_size=2,
    #     max_antecedent_size=3,
    #     max_rules=1,
    # )
    # print("[PRINTING_RULES_FPGROWTH] OK.")

    # print("[PRINTING_RULES_ECLAT]:")
    # print_rules(
    #     eclat_results,
    #     target_consequents={
    #         "response_time_class=not_served",
    #         "response_time_class=fast_le_60s",
    #         "response_time_class=normal_61_180s",
    #         "response_time_class=slow_gt_180s",
    #     },
    #     min_antecedent_size=2,
    #     max_antecedent_size=3,
    #     max_rules=1,
    # )
    # print("[PRINTING_RULES_ECLAT] OK.")
    # print("-" * 100)

    print("[EXTRACT_RULES] Extraindo regras de associação...")
    print("[EXTRACT_RULES_APRIORI]...")
    apriori_rules_df = extract_apriori_rules_to_dataframe(
        apriori_results,
        target_consequents=target_consequents,
        min_antecedent_size=min_antecedent_size,
        max_antecedent_size=max_antecedent_size,
    )
    print("[EXTRACT_RULES_APRIORI] OK.")

    print("[EXTRACT_RULES_FPGROWTH]...")
    fpgrowth_rules_df = extract_fpgrowth_rules_to_dataframe(
        fpgrowth_results,
        target_consequents=target_consequents,
        min_antecedent_size=min_antecedent_size,
        max_antecedent_size=max_antecedent_size,
    )
    print("[EXTRACT_RULES_FPGROWTH] OK.")

    print("[EXTRACT_RULES_ECLAT]...")
    eclat_rules_df = extract_eclat_rules_to_dataframe(
        eclat_results,
        target_consequents=target_consequents,
        min_antecedent_size=min_antecedent_size,
        max_antecedent_size=max_antecedent_size,
    )
    print("[EXTRACT_RULES_ECLAT] OK.")

    experiment_summary_rows.append(
        build_experiment_summary_row(
            dataset_type=type_dataset,
            algorithm_name="Apriori",
            records=records,
            raw_results=apriori_results,
            filtered_rules_df=apriori_rules_df,
            execution_time_s=apriori_time,
            target_consequents=target_consequents,
            min_antecedent_size=min_antecedent_size,
            max_antecedent_size=max_antecedent_size,
        )
    )

    experiment_summary_rows.append(
        build_experiment_summary_row(
            dataset_type=type_dataset,
            algorithm_name="FP-Growth",
            records=records,
            raw_results=fpgrowth_results,
            filtered_rules_df=fpgrowth_rules_df,
            execution_time_s=fpgrowth_time,
            target_consequents=target_consequents,
            min_antecedent_size=min_antecedent_size,
            max_antecedent_size=max_antecedent_size,
        )
    )

    experiment_summary_rows.append(
        build_experiment_summary_row(
            dataset_type=type_dataset,
            algorithm_name="ECLAT",
            records=records,
            raw_results=eclat_results,
            filtered_rules_df=eclat_rules_df,
            execution_time_s=eclat_time,
            target_consequents=target_consequents,
            min_antecedent_size=min_antecedent_size,
            max_antecedent_size=max_antecedent_size,
        )
    )
    print("-" * 100)

    print("[EXPORT_RULES] Exportando regras de associação para CSV...")
    print("[EXPORT_RULES_APRIORI]...")
    export_rules(apriori_rules_df, "outputARM/apriori_observational_output.csv")
    print("[EXPORT_RULES_APRIORI] OK.")

    print("[EXPORT_RULES_FPGROWTH]...")
    export_rules(fpgrowth_rules_df, "outputARM/fpgrowth_observational_output.csv")
    print("[EXPORT_RULES_FPGROWTH] OK.")

    print("[EXPORT_RULES_ECLAT]...")
    export_rules(eclat_rules_df, "outputARM/eclat_observational_output.csv")
    print("[EXPORT_RULES_ECLAT] OK.")

    export_experiment_summary(
        experiment_summary_rows,
        f"outputARM/arm_experiment_summary_{type_dataset}.csv",
    )

    print("-" * 100)

elif type_dataset == "synthetic":
    
    print("-" * 100)
    print(f"[DATASET] Tipo do dataset: {type_dataset}")

    print("[CLEANING] Executando pipeline de limpeza...")
    cleaning_preprocess_pipeline()
    print("[CLEANING] OK.")
    print("-" * 100)

    print("[SYNTHETIC_DATASET_EXPANSION] Executando pipeline de expansão do dataset sintético...")
    synthetic_dataset_expansion_pipeline()
    print("[SYNTHETIC_DATASET_EXPANSION] OK.")
    print("-" * 100)

    print("[FEATURE_ENGINEERING] Executando pipeline de engenharia de features...")
    feature_engineering_pipeline(type_dataset="synthetic")
    print("[FEATURE_ENGINEERING] OK.")
    print("-" * 100)

    print("[DISCRETIZATION_CATEGORICAL] Executando pipeline de discretização categórica...")
    discretization_categorical_pipeline(type_dataset="synthetic")
    print("[DISCRETIZATION_CATEGORICAL] OK.")
    print("-" * 100)

    print("[TRANSACTIONAL_FORMAT] Executando pipeline para formato transacional...")
    records = transactional_format_pipeline(type_dataset="synthetic")
    print("[TRANSACTIONAL_FORMAT] OK.")
    print("-" * 100)

    print("[ARM] Executando pipelines ARM...")
    apriori_results, apriori_time = run_with_timer(
        "APRIORI",
        apriori_pipeline,
        records,
    )
    print("[ARM_APRIORI] OK.")
    fpgrowth_results, fpgrowth_time = run_with_timer(
        "FP-GROWTH",
        fpgrowth_pipeline,
        records,
    )
    print("[ARM_FPGROWTH] OK.")
    eclat_results, eclat_time = run_with_timer(
        "ECLAT",
        eclat_pipeline,
        records,
    )
    print("[ARM_ECLAT] OK.")
    print("-" * 100)

    # print("[PRINTING_RULES] Printando regras de associação...")
    # print("[PRINTING_RULES_APRIORI]:")
    # print_rules(
    #     apriori_results,
    #     target_consequents={
    #         "response_time_class=not_served",
    #         "response_time_class=fast_le_60s",
    #         "response_time_class=normal_61_180s",
    #         "response_time_class=slow_gt_180s",
    #     },
    #     min_antecedent_size=2,
    #     max_antecedent_size=3,
    #     max_rules=1,
    # )
    # print("[PRINTING_RULES_APRIORI] OK.")

    # print("[PRINTING_RULES_FPGROWTH]:")
    # print_rules(
    #     fpgrowth_results,
    #     target_consequents={
    #         "response_time_class=not_served",
    #         "response_time_class=fast_le_60s",
    #         "response_time_class=normal_61_180s",
    #         "response_time_class=slow_gt_180s",
    #     },
    #     min_antecedent_size=2,
    #     max_antecedent_size=3,
    #     max_rules=1,
    # )
    # print("[PRINTING_RULES_FPGROWTH] OK.")

    # print("[PRINTING_RULES_ECLAT]:")
    # print_rules(
    #     eclat_results,
    #     target_consequents={
    #         "response_time_class=not_served",
    #         "response_time_class=fast_le_60s",
    #         "response_time_class=normal_61_180s",
    #         "response_time_class=slow_gt_180s",
    #     },
    #     min_antecedent_size=2,
    #     max_antecedent_size=3,
    #     max_rules=1,
    # )
    # print("[PRINTING_RULES_ECLAT] OK.")
    # print("-" * 100)

    print("[EXTRACT_RULES] Extraindo regras de associação...")
    print("[EXTRACT_RULES_APRIORI]...")
    apriori_rules_df = extract_apriori_rules_to_dataframe(
        apriori_results,
        target_consequents=target_consequents,
        min_antecedent_size=min_antecedent_size,
        max_antecedent_size=max_antecedent_size,
    )
    print("[EXTRACT_RULES_APRIORI] OK.")

    print("[EXTRACT_RULES_FPGROWTH]...")
    fpgrowth_rules_df = extract_fpgrowth_rules_to_dataframe(
        fpgrowth_results,
        target_consequents=target_consequents,
        min_antecedent_size=min_antecedent_size,
        max_antecedent_size=max_antecedent_size,
    )
    print("[EXTRACT_RULES_FPGROWTH] OK.")

    print("[EXTRACT_RULES_ECLAT]...")
    eclat_rules_df = extract_eclat_rules_to_dataframe(
        eclat_results,
        target_consequents=target_consequents,
        min_antecedent_size=min_antecedent_size,
        max_antecedent_size=max_antecedent_size,
    )
    print("[EXTRACT_RULES_ECLAT] OK.")

    experiment_summary_rows.append(
        build_experiment_summary_row(
            dataset_type=type_dataset,
            algorithm_name="Apriori",
            records=records,
            raw_results=apriori_results,
            filtered_rules_df=apriori_rules_df,
            execution_time_s=apriori_time,
            target_consequents=target_consequents,
            min_antecedent_size=min_antecedent_size,
            max_antecedent_size=max_antecedent_size,
        )
    )

    experiment_summary_rows.append(
        build_experiment_summary_row(
            dataset_type=type_dataset,
            algorithm_name="FP-Growth",
            records=records,
            raw_results=fpgrowth_results,
            filtered_rules_df=fpgrowth_rules_df,
            execution_time_s=fpgrowth_time,
            target_consequents=target_consequents,
            min_antecedent_size=min_antecedent_size,
            max_antecedent_size=max_antecedent_size,
        )
    )

    experiment_summary_rows.append(
        build_experiment_summary_row(
            dataset_type=type_dataset,
            algorithm_name="ECLAT",
            records=records,
            raw_results=eclat_results,
            filtered_rules_df=eclat_rules_df,
            execution_time_s=eclat_time,
            target_consequents=target_consequents,
            min_antecedent_size=min_antecedent_size,
            max_antecedent_size=max_antecedent_size,
        )
    )

    print("-" * 100)

    print("[EXPORT_RULES] Exportando regras de associação para CSV...")
    print("[EXPORT_RULES_APRIORI]...")
    export_rules(apriori_rules_df, "outputARM/apriori_synthetic_output.csv")
    print("[EXPORT_RULES_APRIORI] OK.")

    print("[EXPORT_RULES_FPGROWTH]...")
    export_rules(fpgrowth_rules_df, "outputARM/fpgrowth_synthetic_output.csv")
    print("[EXPORT_RULES_FPGROWTH] OK.")

    print("[EXPORT_RULES_ECLAT]...")
    export_rules(eclat_rules_df, "outputARM/eclat_synthetic_output.csv")
    print("[EXPORT_RULES_ECLAT] OK.")

    export_experiment_summary(
        experiment_summary_rows,
        f"outputARM/arm_experiment_summary_{type_dataset}.csv",
    )

    print("-" * 100)



elif step in [
    "visualizacao_distribuicao_tempo_resposta",
    "visualizacao_boxplot_tempo_resposta",
    "visualizacao_tempo_resposta_por_hora",
    "visualizacao_taxa_atendimento_dia_util_vs_fim_de_semana",
    "visualizacao_matriz_correlacao",
    "visualizacao_todas",
    "dataset_quality_summary",
    "response_time_statistics",
    "boxplot_outlier_statistics",
    "service_rate_by_hour",
    "service_rate_weekday_vs_weekend",
    "correlation_statistics",
    "run_all_statistics"
]:
    from eda import (
        visualize_response_time_distribution, 
        visualize_response_time_boxplot, 
        visualize_response_time_by_hour,
        visualize_weekend_vs_weekday_response_rate,
        visualize_correlation_matrix,
        visualize_all,
        dataset_quality_summary,
        response_time_statistics,
        boxplot_outlier_statistics,
        service_rate_by_hour,
        service_rate_weekday_vs_weekend,
        correlation_statistics,
        run_all_statistics
    ) 

    if step == "visualizacao_distribuicao_tempo_resposta":
        visualize_response_time_distribution()
    
    elif step == "visualizacao_boxplot_tempo_resposta":
        visualize_response_time_boxplot()

    elif step == "visualizacao_tempo_resposta_por_hora":
        visualize_response_time_by_hour()

    elif step == "visualizacao_taxa_atendimento_dia_util_vs_fim_de_semana":
        visualize_weekend_vs_weekday_response_rate()

    elif step == "visualizacao_matriz_correlacao":
        visualize_correlation_matrix()

    elif step == "visualizacao_todas":
        visualize_all()

    elif step == "dataset_quality_summary":
        dataset_quality_summary()

    elif step == "response_time_statistics":
        response_time_statistics()

    elif step == "boxplot_outlier_statistics":
        boxplot_outlier_statistics()
    
    elif step == "service_rate_by_hour":
        service_rate_by_hour()

    elif step == "service_rate_weekday_vs_weekend":
        service_rate_weekday_vs_weekend()

    elif step == "correlation_statistics": 
        correlation_statistics()

    elif step == "run_all_statistics":
        run_all_statistics()



