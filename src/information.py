from src.dataset import TUSimpleDataset

data_path = "../data/TUSimple/train_set/"

json_mapping = {
    '0313-1': 'label_data_0313.json',
    '0313-2': 'label_data_0313.json',
    '0531': 'label_data_0531.json',
    '0601': 'label_data_0601.json'
}

train_folders = ['0313-1', '0313-2']
val_folders = ['0531']
test_folders = ['0601']
train_dataset = TUSimpleDataset(data_path, json_mapping, train_folders, target_height=360)
val_dataset = TUSimpleDataset(data_path, json_mapping, val_folders, target_height=360)
test_dataset = TUSimpleDataset(data_path, json_mapping, test_folders, target_height=360)
