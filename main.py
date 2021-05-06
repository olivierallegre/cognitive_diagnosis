from knowledge_tracing_models.process_data import knowledge_tracing_processing
from factor_analysis_models.process_data import factor_analysis_processing
import pandas as pd

if __name__ == '__main__':
    data = pd.read_csv('data/data.csv').head(200000)
    results, errors = knowledge_tracing_processing(dataset=data, model='kt-bkt')
    print(results)
    results, errors = factor_analysis_processing(dataset=data, model='irt', d=5)
    print(results)
