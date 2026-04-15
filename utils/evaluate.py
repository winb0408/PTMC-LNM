import os
import pandas as pd

# 定义文件夹路径
folder_path = '/data2/wjb/code/PTMC_LNM/results/SwinTransformer/finetune500//class/'  # 替换为你的文件夹路径
output_folder = '/data2/wjb/code/PTMC_LNM/results/SwinTransformer/finetune500/'  # 替换为你的输出文件夹路径

# 确保输出文件夹存在
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 定义输出文件路径
output_csv_path = os.path.join(output_folder, 'results_summary.csv')
output_excel_path = os.path.join(output_folder, 'results_summary.xlsx')

# 定义一个空的DataFrame来存储结果
results_df = pd.DataFrame(columns=[
    'Image Name',
    # 'Ki67 GT', 'Ki67 Pred',
    # 'LNM GT', 'LNM Pred',
    'Central LNM GT', 'Central LNM Pred'
])

# 遍历文件夹中的所有txt文件
for filename in os.listdir(folder_path):
    if filename.endswith('.txt'):
        file_path = os.path.join(folder_path, filename)

        # 初始化一个字典来存储当前文件的结果
        result_dict = {'Image Name': filename}

        # 读取文件内容
        with open(file_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                key, value = line.strip().split(': ')
                # 根据不同的任务提取Ground Truth和Prediction
                if key == 'Image Name':
                    result_dict[key] = value
                elif key == 'Ground Truth Central LNM':
                    result_dict['Central LNM GT'] = int(value)
                elif key == 'Prediction Central LNM':
                    result_dict['Central LNM Pred'] = int(value)
                elif key == 'Ground Truth':
                    result_dict['Central LNM GT'] = int(value)
                elif key == 'Prediction':
                    result_dict['Central LNM Pred'] = int(value)

        # 将结果添加到DataFrame中
        results_df = pd.concat([results_df, pd.DataFrame([result_dict])], ignore_index=True)

# 保存汇总表格到CSV文件
results_df.to_csv(output_csv_path, index=False)
print(f"CSV file saved to {output_csv_path}")

# 保存汇总表格到Excel文件
results_df.to_excel(output_excel_path, index=False)
print(f"Excel file saved to {output_excel_path}")

# 计算TP、FP、TN、FN
tp_central_lnm = ((results_df['Central LNM GT'] == 1) & (results_df['Central LNM Pred'] == 1)).sum()
fp_central_lnm = ((results_df['Central LNM GT'] == 0) & (results_df['Central LNM Pred'] == 1)).sum()
tn_central_lnm = ((results_df['Central LNM GT'] == 0) & (results_df['Central LNM Pred'] == 0)).sum()
fn_central_lnm = ((results_df['Central LNM GT'] == 1) & (results_df['Central LNM Pred'] == 0)).sum()

# 计算ACC、SEN、SPE
acc_central_lnm = (tp_central_lnm + tn_central_lnm) / (
            tp_central_lnm + fp_central_lnm + tn_central_lnm + fn_central_lnm)
sen_central_lnm = tp_central_lnm / (tp_central_lnm + fn_central_lnm)
spe_central_lnm = tn_central_lnm / (tn_central_lnm + fp_central_lnm)

# 打印结果
print("\nTP, FP, TN, FN：")
print(f"Central LNM: TP={tp_central_lnm}, FP={fp_central_lnm}, TN={tn_central_lnm}, FN={fn_central_lnm}")

print("\nACC, SEN, SPE：")
print(f"Central LNM: ACC={acc_central_lnm:.4f}, SEN={sen_central_lnm:.4f}, SPE={spe_central_lnm:.4f}")















# import numpy as np
# import matplotlib.pyplot as plt
#
# # 替换这里的数据为你的混淆矩阵
# conf_matrix = np.array([[75, 17], [81, 26]])
# TN = conf_matrix[0, 0]
# FP = conf_matrix[0, 1]
# FN = conf_matrix[1, 0]
# TP = conf_matrix[1, 1]
#
# # 计算评估指标
# accuracy = (TN + TP) / np.sum(conf_matrix)
# recall = TP / (TP + FN)
# precision = TP / (TP + FP)
# f1 = 2 * precision * recall / (precision + recall)
#
# # 计算灵敏度和特异度
# sensitivity = TP / (TP + FN)
# specificity = TN / (FP + TN)
#
# # 计算平衡精度
# balanced_accuracy = (sensitivity + specificity) / 2
#
# # 输出指标
# print(f"Accuracy: {accuracy:.4f}")
# print(f"Recall: {recall:.4f}")
# print(f"Precision: {precision:.4f}")
# print(f"F1 Score: {f1:.4f}")
# print(f"Sensitivity: {sensitivity:.4f}")
# print(f"Specificity: {specificity:.4f}")
# print(f"Balanced Accuracy: {balanced_accuracy:.4f}")
#
# # 绘制混淆矩阵图
# labels = ['Class 0', 'Class 1']
# cmap = plt.cm.Blues
# fig, ax = plt.subplots()
# im = ax.imshow(conf_matrix, interpolation='nearest', cmap=cmap)
#
# # 添加颜色和调整字号
# for i in range(len(labels)):
#     for j in range(len(labels)):
#         color = 'black'  # 数字颜色为黑色
#         if i == j:
#             color = (208/255, 172/255, 196/255)  # R208 G172 B196
#         elif i + j == len(labels) - 1:
#             color = (237/255, 224/255, 228/255)  # R237 G224 B228
#         plt.text(j, i, str(conf_matrix[i, j]), color=color, ha='center', va='center', fontsize=14)
#
# plt.title('Confusion Matrix', fontsize=16)
# plt.colorbar(im)
# tick_marks = np.arange(len(labels))
# plt.xticks(tick_marks, labels, fontsize=14)
# plt.yticks(tick_marks, labels, fontsize=14)
#
# plt.ylabel('True label', fontsize=16)
# plt.xlabel('Predicted label', fontsize=16)
# plt.show()