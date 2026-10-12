import pandas as pd


csv_file = '/home/24071213138/MeModel/datasets/Neuro_dataset/testing.csv'
df = pd.read_csv(csv_file, header=0)  # header=0表示第一行作为表头

def save_as_fasta(df, filename, prefix):
    with open(filename, 'w') as f:
        # 重置索引，使ID从0开始
        df_reset = df.reset_index(drop=True)
        for idx, row in df_reset.iterrows():
            f.write(f">{prefix}_{idx}\n{row['sequence']}\n")

pos_df = df[df['label'] == 1]
neg_df = df[df['label'] == 0]

# 保存为 FASTA 文件（负样本ID从0开始）
save_as_fasta(pos_df, './test_pos.fasta', 'Positive')
save_as_fasta(neg_df, './test_neg.fasta', 'Negative')

print("FASTA files have been created successfully.")
