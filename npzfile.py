import numpy as np

# 从npz文件加载数组
data = np.load(r'G:\CAFE\dataset\Twitter_Rumor_Detection\train_image_with_label.npz')

# 查看文件中保存的数组名字
print(data.files)

# 获取特定的数组
array1 = data['data']
array2 = data['label']

print(array1.shape)
# print(array1)
# print(array2)
# 使用加载的数组进行操作
# ...

# 关闭文件
data.close()