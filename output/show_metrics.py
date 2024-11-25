import pandas as pd
import matplotlib.pyplot as plt

file_path_1 = '//model_metrics/alpha_mfcnn_Set_2_0_1_005.csv'
file_path_2 = '//model_metrics/alpha_cxn_Set_2_09_1_001.csv'
file_path_3 = '//model_metrics/alpha_cloudfcn_Set_2_09_1_001.csv'

data1 = pd.read_csv(file_path_1)
data2 = pd.read_csv(file_path_2)
data3 = pd.read_csv(file_path_3)

colors = {
    'Accuracy': 'blue',
    'Precision': 'green',
    'Recall': 'red',
    'F1 Score': 'orange'
}

plt.figure(figsize=(15, 10))

# MFCNN
plt.plot(data1['Alpha'], data1['Accuracy'], label='Accuracy (MFCNN)', color=colors['Accuracy'], marker='o')
plt.plot(data1['Alpha'], data1['Precision'], label='Precision (MFCNN)', color=colors['Precision'], marker='s')
plt.plot(data1['Alpha'], data1['Recall'], label='Recall (MFCNN)', color=colors['Recall'], marker='^')
plt.plot(data1['Alpha'], data1['F1 Score'], label='F1 Score (MFCNN)', color=colors['F1 Score'], marker='d')

# CXN
plt.plot(data2['Alpha'], data2['Accuracy'], label='Accuracy (CXN)', color=colors['Accuracy'], linestyle='--', marker='o')
plt.plot(data2['Alpha'], data2['Precision'], label='Precision (CXN)', color=colors['Precision'], linestyle='--', marker='s')
plt.plot(data2['Alpha'], data2['Recall'], label='Recall (CXN)', color=colors['Recall'], linestyle='--', marker='^')
plt.plot(data2['Alpha'], data2['F1 Score'], label='F1 Score (CXN)', color=colors['F1 Score'], linestyle='--', marker='d')

# # CLOUDFCN
# plt.plot(data3['Alpha'], data3['Accuracy'], label='Accuracy (CLOUDFCN)', color=colors['Accuracy'], linestyle='-.', marker='o')
# plt.plot(data3['Alpha'], data3['Precision'], label='Precision (CLOUDFCN)', color=colors['Precision'], linestyle='-.', marker='s')
# plt.plot(data3['Alpha'], data3['Recall'], label='Recall (CLOUDFCN)', color=colors['Recall'], linestyle='-.', marker='^')
# plt.plot(data3['Alpha'], data3['F1 Score'], label='F1 Score (CLOUDFCN)', color=colors['F1 Score'], linestyle='-.', marker='d')

plt.title('Comparison of Metrics MFCNN_CXN_CLOUDFCN')
plt.xlabel('Alpha')
plt.ylabel('Metric Value')
plt.ylim(0, 1.1)
plt.legend()

# Показать график
plt.grid(True)
plt.show()
