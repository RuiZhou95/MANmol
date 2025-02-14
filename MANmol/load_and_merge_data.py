import pandas as pd
import pickle

def load_and_merge_data(ID_smiles_graph_path, properties_path, chunk_size=1000):
    properties_df = pd.read_csv(properties_path, index_col='ID').drop(columns=['smiles'], errors='ignore')
    merged_data = []

    with open(ID_smiles_graph_path, 'rb') as f:
        while True:
            try:
                all_data = pickle.load(f)
                for item in all_data:
                    if item['ID'] in properties_df.index:
                        properties = properties_df.loc[item['ID']].to_dict()
                        merged_entry = {**item, **properties}
                        merged_data.append(merged_entry)
                break
            except EOFError:
                break 
            except Exception as e:
                print(f"Error: {e}")
                break

    return merged_data

ID_smiles_graph_path = './data/fine_tuning_data/all_data_ID-smiles-graph-test.pkl'
properties_path = './data/all_porperty_AE_data-test.csv'
merged_data = load_and_merge_data(ID_smiles_graph_path, properties_path, chunk_size=1000)
