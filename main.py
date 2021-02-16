from knowledge_tracing_models.process_data import process_dataset
import pandas as pd

if __name__ == '__main__':
    data = pd.read_csv('data/data.csv').head(200000)
    results, errors = process_dataset(dataset=data, model='kt-bkt', d=5)
    print(results['auc'])
