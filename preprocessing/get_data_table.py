import pandas as pd
import random

# SAPdb数据集
df = pd.read_csv('./datas/SAPdb_ps.csv',encoding='ISO-8859-1')

print(len(df))

# 筛选出'column_name'不为某些项的数据
# 'Twisted fibers'
excluded_items = ['Unstable hydrogel assembly','None','NA','No self assembled structure','No aggregation','Disordered structure','Disorder Short fibres','Suspension','Solution','No hydrogel formation',
                  'Amorphous aggregate','No gel formation','No hydrogel','Unstable hydrogel','No Nanoparticle formation','Precipitation','No organogel formation','Irregular rod like aggregate structure',
                  'Irregular rod-like aggregates as well as plate-like structures','Irregular plates','nan',
                  'Irregular rod-like structures','Rod like Nanostructure','No hydogel formation','No assembled structure','No hydrogel formation or unstable hydrogel','no self-assembly','precipitated','bead like nanostructure','turbid gel (consists of fibers)','turbid gel (containing nanospheres)',
                  'Shrank particles','Porous gel consists of irregular fibers','Randomly ordered, dense tubular nanostructures','Opaque hydrogel','Opaque gel','Turbid gel','Fibers (untwisted) like precipitates','Unstable hydrogel or small aggregate structure',
                  'small aggregate structure','opaque hydrogel gel (nanofibers)','unstable hyrogel or small aggregate structure']

df['TYPE OF SELF-ASSEMBLY'] = (
    df['TYPE OF SELF-ASSEMBLY']
    .astype(str)                      # 确保列是字符串类型
    .str.strip()                      # 去掉首尾空格
    .str.replace(r'\s+', ' ', regex=True)  # 清理多余空格
    .str.lower()                      # 统一为小写
)

excluded_items = [item.strip().lower() for item in excluded_items]

# 自定义过滤条件
filtered_df = df[df['TYPE OF SELF-ASSEMBLY'].apply(lambda x: x not in excluded_items)]

# 获取 'TYPE OF SELF-ASSEMBLY' 列的唯一值
unique_values = filtered_df['TYPE OF SELF-ASSEMBLY'].unique()

# 将唯一值转换为 DataFrame
unique_df = pd.DataFrame(unique_values, columns=['TYPE OF SELF-ASSEMBLY'])

# 保存为 CSV 文件
unique_df.to_csv('./datas/unique_self_assembly_types.csv', index=False)

print(filtered_df['TYPE OF SELF-ASSEMBLY'].unique())

filtered_df = filtered_df.dropna(subset=['PEPTIDE SEQUENCE'])
df_filtered = filtered_df[filtered_df['PEPTIDE SEQUENCE'].str.strip().str.match('^[a-zA-Z]+$')]
df_filtered = df_filtered[df_filtered['PEPTIDE SEQUENCE'].str.len() == 3]

# 统一转为大写并获取唯一值
unique_values = list(df_filtered['PEPTIDE SEQUENCE'].str.upper().unique())
print(len(unique_values))
print(unique_values)

df = df.dropna(subset=['PEPTIDE SEQUENCE'])
# 筛选：只保留字母，长度小于3，并且不在unique_values中
filtered_df = df[df['PEPTIDE SEQUENCE'].str.match('^[a-zA-Z]+$')]  # 只包含字母

filtered_df = filtered_df[filtered_df['PEPTIDE SEQUENCE'].str.len() == 3]  # 长度小于3
filtered_df = filtered_df[~filtered_df['PEPTIDE SEQUENCE'].str.upper().isin(unique_values)]  # 不在unique_values中

# 统一转换为小写并获取唯一值
unique_filtered_values = list(filtered_df['PEPTIDE SEQUENCE'].str.upper().str.strip().unique())
print(unique_filtered_values)

print("dataset1 ava:",len(unique_filtered_values)+len(unique_values))

common_elements = set(unique_values) & set(unique_filtered_values)

# 判断是否有共同元素
if common_elements:
    print(f"共同元素: {common_elements}")
else:
    print("没有共同元素")

#爬取的数据集
df2 = pd.read_csv('./datas/phase_data_clean.csv',encoding='ISO-8859-1')

print("dataset2 total:",len(df2))

data_t = df2[df2['type'] != 0]['sequence']
data_t = data_t[data_t.str.len()==3]
unique_seq_t = list(data_t.str.upper().str.strip().unique())
print(unique_seq_t)

data_f = df2[~df2['sequence'].str.upper().isin(unique_seq_t)]['sequence']
data_f = data_f[data_f.str.len()==3]
unique_seq_f = list(data_f.str.upper().str.strip().unique())
print(unique_seq_f)
print("dataset2 ava:",len(unique_seq_f)+len(unique_seq_t))

common_elements = set(unique_seq_t) & set(unique_seq_f)

# 判断是否有共同元素
if common_elements:
    print(f"共同元素: {common_elements}")
else:
    print("没有共同元素")

common_elements = set(unique_values) & set(unique_seq_f)

# 判断是否有共同元素
if common_elements:
    print(f"共同元素: {common_elements}")
else:
    print("没有共同元素")

df3 = pd.read_excel('./datas/1.xlsx',sheet_name='Self-assembling sequences')

print("pos：",len(df3))

print(len(df3[df3['Peptide sequence (one letter code)'].str.len()==4]))

print(df3[df3['Peptide sequence (one letter code)'].str.len()==10])
df_l10 = df3[df3['Peptide sequence (one letter code)'].str.len()==10]

print(df_l10['Peptide sequence (one letter code)'].tolist())

df_l10.to_csv('./datas/l10_self_assembly.csv', index=False)

df3 = df3[df3['Peptide sequence (one letter code)'].str.len()<=15]
data_t_2 = list(df3['Peptide sequence (one letter code)'].str.upper().unique())


df5_t = df3[((df3['Peptide sequence (one letter code)'].str.len()<=15)&(df3['Peptide sequence (one letter code)'].str.len()>=10))]

print(df5_t['Peptide sequence (one letter code)'].tolist())

ll10_list = df5_t['Peptide sequence (one letter code)'].tolist()
label_t = [1]*len(ll10_list)

df_sl10 = pd.DataFrame({
    'sequence': ll10_list,
    'label': label_t
})

df_sl10.to_csv('./datas/l10_self_assembly.csv', index=False)

print(data_t_2)

df4 = pd.read_excel('./datas/1.xlsx',sheet_name='Non-assembling sequences')
print(len(df4[df4['Peptide sequence (one letter code)'].str.len()==3]))

print("neg:",len(df4))

df5 = df4[((df4['Peptide sequence (one letter code)'].str.len()<=15)&(df4['Peptide sequence (one letter code)'].str.len()>10))]

print(df5['Peptide sequence (one letter code)'].tolist())

df4 = df4[df4['Peptide sequence (one letter code)'].str.len()<=15]
data_f_2 = list(df4['Peptide sequence (one letter code)'].str.upper().unique())
print(data_f_2)

print("dataset3 total:",len(df3)+len(df4))

common_elements = set(data_t_2) & set(data_f_2)

# 判断是否有共同元素
if common_elements:
    print(f"共同元素: {common_elements}")
else:
    print("没有共同元素")

merge_t_list = list(set(data_t_2 + unique_seq_t + unique_values))
print(merge_t_list)

merge_f_list = list(set(data_f_2 + unique_seq_f + unique_filtered_values))
print(merge_f_list)

common_elements = set(merge_f_list) & set(merge_t_list)

merge_f_list = [item for item in merge_f_list if item not in common_elements]
merge_t_list = [item for item in merge_t_list if item not in common_elements]

# 判断是否有共同元素
if common_elements:
    print(f"共同元素: {common_elements}")
else:
    print("没有共同元素")

label_t = [1]*len(merge_t_list)
label_f = [0]*len(merge_f_list)

data = merge_t_list + merge_f_list
label = label_t + label_f

df_s = pd.DataFrame({
    'sequence': data,
    'label': label
})

amino_acids = list('ACDEFGHIKLMNPQRSTVWY')

def replace_Z_random(seq):
    return ''.join([random.choice(amino_acids) if aa == 'Z' else aa for aa in seq])

df_s['sequence'] = df_s['sequence'].apply(replace_Z_random)
df_s = df_s[df_s['sequence'].str.len() >=10]
# 2. 删除包含 'X' 的序列
df_s = df_s[~df_s['sequence'].str.contains('X')].reset_index(drop=True)

df_s.to_csv('./datas/merge_dataset_temp.csv', index=False)
