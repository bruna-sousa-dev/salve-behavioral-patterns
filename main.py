from pipelines import (
    deterministic_pipeline
)
from visualization import (
    visualize_response_time_distribution, 
    visualize_response_time_boxplot, 
    visualize_response_time_by_hour,
    visualize_weekend_vs_weekday_response_rate,
    visualize_correlation_matrix,
    visualize_all
) 

"""
Definir qual pipeline executar
Opções: 
-> "pipeline_deterministico": 
      Executa a pipeline determinística completa, incluindo todas as etapas de pré-processamento, 
      análise e visualização.

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
"""
step = "visualizacao_todas"


if __name__ == "__main__":
    
    if step == "pipeline_deterministico":
        deterministic_pipeline()

    elif step == "visualizacao_distribuicao_tempo_resposta":
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
    
