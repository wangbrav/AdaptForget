import matplotlib.pyplot as plt
epochs = []  # Epoch编号
p_values = []  # 对应的p-value
accuracies = []  # 对应的准确率

def update_plot_with_new_data(new_epoch, new_p_value, new_accuracy):
    """
    将新的epoch、对应的p-value和准确率添加到折线图中，并重新绘制图表。

    参数:
    - new_epoch: 新的epoch值（整数）。
    - new_p_value: 对应于新epoch的p-value（浮点数）。
    - new_accuracy: 对应于新epoch的准确率（浮点数或整数）。
    """
    # 将新的数据点添加到全局变量中
    epochs.append(new_epoch)
    p_values.append(new_p_value)
    accuracies.append(new_accuracy)

    # 创建图表
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # 绘制p-value折线图
    color = 'tab:blue'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('P-value', color=color)
    ax1.plot(epochs, p_values, marker='o', linestyle='-', color=color, label='P-value')
    ax1.tick_params(axis='y', labelcolor=color)
    # ax1.set_ylim([0, 0.025])  # 例如，设置p-value的范围为0到1

    # 创建共享x轴的第二个y轴，用于绘制准确率
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Accuracy', color=color)
    ax2.plot(epochs, accuracies, marker='x', linestyle='--', color=color, label='Accuracy')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim([0, 100])  # 例如，设置准确率的范围为0%到100%

    # 添加图表标题和网格线
    plt.title('Epoch vs. P-value and Accuracy')
    fig.tight_layout()  # 调整布局以防止重叠
    plt.grid(True)

    # 显示图表
    plt.show()
