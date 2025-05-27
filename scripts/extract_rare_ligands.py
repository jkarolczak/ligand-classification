import os
import pandas as pd

from cfg import read_config

if __name__ == '__main__':
    cfg = read_config("../cfg/rare_ligands.yaml")
    files_ligands = cfg["data_paths"]
    class_mapping = pd.read_csv(cfg["class_mapping"])
    class_mapping = {k: v for k, v in zip(class_mapping["ligand"], class_mapping["cluster_name"])}
    dfs = []
    for f in files_ligands:
        dfs.append(pd.read_csv(f))
    df = pd.concat(dfs, ignore_index=True)

    rare_only_df = df[df["ligand"] == "RARE_LIGAND"]
    files_main_dir = os.listdir(cfg['input_dir'])
    rare_only_df = rare_only_df[rare_only_df["blob_map_filename"].isin(files_main_dir)]
    rare_only_df['data'] = [cfg['data_type']] * len(rare_only_df)
    rare_only_df['dir'] = [os.path.basename(cfg["rare_blobs_dir"])] * len(rare_only_df)
    if cfg['data_type'] == 'xray':
        rare_only_df['type'] = rare_only_df['blob_map_filename'].apply(lambda x: x.split("_")[1])
    else:
        rare_only_df['type'] = rare_only_df['blob_map_filename'].apply(lambda x: x.split("_")[-1].split(".")[0])
    rare_only_df['class'] = [
        class_mapping[rare_only_df['type'].iloc[i]] if rare_only_df['type'].iloc[i] in class_mapping else
        rare_only_df['type'].iloc[i] for i in range(len(rare_only_df))]
    rare_only_df.to_csv(cfg['output_filename'], index=False)

    files = rare_only_df["blob_map_filename"].tolist()
    os.makedirs(cfg['rare_blobs_dir'], exist_ok=True)
    files_main_dir = os.listdir(cfg['input_dir'])
    files_difference = set(files).difference(set(files_main_dir))
    files_common = set(files).intersection(set(files_main_dir))
    print(files_difference, len(files), len(files_difference), len(files_common))
    for f in files_common:
        os.system(f"cp --preserve=all {os.path.join(cfg['input_dir'], f)} {cfg['rare_blobs_dir']}")

    # cryoem_ligands = pd.read_csv("../data/cryoem_qscores.csv")
    # dfs = []
    # files_ligands = ['../data/xray_holdout.csv', '../data/xray_train.csv']
    # for f in files_ligands:
    #     dfs.append(pd.read_csv(f))
    # all_classes = pd.concat(dfs, ignore_index=True)
    # all_classes = all_classes['ligand'].unique().tolist()
    # all_classes = [c for c in all_classes if c != 'RARE_LIGAND']
    # print(len(all_classes))
    # ligand_mapping = pd.read_csv("../data/ligand_mapping.csv")
    # ligand_mapping = {k: v for k, v in zip(ligand_mapping["ligand"], ligand_mapping["cluster_name"])}
    # cryoem_ligands['file_type'] = cryoem_ligands['id'].apply(lambda x: x.split("_")[-1].split(".")[0])
    # classes = []
    # for i in range(len(cryoem_ligands)):
    #     f_type = cryoem_ligands['file_type'].iloc[i]
    #     ligand_map = ligand_mapping[f_type] if f_type in ligand_mapping else f_type
    #     cls = ligand_map if ligand_map in all_classes else 'RARE_LIGAND'
    #     classes.append(cls)
    # cryoem_ligands['ligand'] = classes
    # cryoem_ligands.rename(columns={'id': 'blob_map_filename'}, inplace=True)
    # cryoem_ligands.to_csv("../data/cryoem_class.csv", index=False)

