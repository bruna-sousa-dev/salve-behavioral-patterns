step = "synthetic_dataset_expansion_pipeline"
"""
Definir qual pipeline executar
Opções: 
>>> Preparação do dataset:
-> Etapa 1: 
    "cleaning_preproccess_pipeline": 
        Executa a pipeline de limpeza e pré-processamento dos dados.

-> Etapa 2: 
    "synthetic_dataset_expansion_pipeline":
        Executa a pipeline de expansão do dataset usando técnicas de geração de dados sintéticos.   

-> Etapa 3:
    "observational_feature_engineering_pipeline":
        Executa a pipeline de engenharia de features no dataset observacional.
    "syntetic_feature_engineering_pipeline":
        Executa a pipeline de engenharia de features no dataset sintético.    

-> Etapa 4:
    "observational_discretization_categorical_pipeline":
        Executa a pipeline de discretização no dataset observacional.
    "syntetic_discretization_categorical_pipeline":
        Executa a pipeline de discretização no dataset sintético.

-> Etapa 5:
    "observational_transactional_format_pipeline":
        Executa a pipeline de conversão para formato transacional no dataset observacional.
    "synthetic_transactional_format_pipeline":  
        Executa a pipeline de conversão para formato transacional no dataset sintético.


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

if __name__ == "__main__":
    
    if step in [
        "cleaning_preproccess_pipeline", 
        "synthetic_dataset_expansion_pipeline",
        "observational_feature_engineering_pipeline", 
        "synthetic_feature_engineering_pipeline",
        "observational_discretization_categorical_pipeline",
        "synthetic_discretization_categorical_pipeline",
        "observational_transactional_format_pipeline",
        "synthetic_transactional_format_pipeline",
    ]:
        if step == "cleaning_preproccess_pipeline":
            from pipelines import cleaning_preprocess_pipeline
            cleaning_preprocess_pipeline()

        if step == "synthetic_dataset_expansion_pipeline":
            from pipelines import synthetic_dataset_expansion_pipeline
            synthetic_dataset_expansion_pipeline()

        elif step == "observational_feature_engineering_pipeline":
            from pipelines import feature_engineering_pipeline
            feature_engineering_pipeline(type_dataset="observational")
        elif step == "syntetic_feature_engineering_pipeline":
            from pipelines import feature_engineering_pipeline
            feature_engineering_pipeline(type_dataset="synthetic")

        elif step == "observational_discretization_categorical_pipeline":
            from pipelines import discretization_categorical_pipeline
            discretization_categorical_pipeline(type_dataset="observational")
        elif step == "syntetic_discretization_categorical_pipeline":
            from pipelines import discretization_categorical_pipeline
            discretization_categorical_pipeline(type_dataset="synthetic")

        elif step == "observational_transactional_format_pipeline":
            from pipelines import transactional_format_pipeline
            transactional_format_pipeline(type_dataset="observational")
        elif step == "synthetic_transactional_format_pipeline":
            from pipelines import transactional_format_pipeline
            transactional_format_pipeline(type_dataset="synthetic")



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
    
    

