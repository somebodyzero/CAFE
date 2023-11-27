import torch
import torchvision.transforms as transforms
from torchvision.models import resnet34
import json
from PIL import Image
from transformers import BertTokenizer, BertModel

# 调用BERT模型对JSON文件进行处理
def process_json(json_path):
    # 加载BERT模型和tokenizer
    model_name = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)

    # 读取JSON文件
    with open(json_path, 'r') as f:
        data = json.load(f)

    # 提取文本内容
    text = data['text']

    # 使用tokenizer对文本进行编码
    encoded_input = tokenizer.encode_plus(text, add_special_tokens=True, padding='max_length', max_length=512, return_tensors='pt')

    # 使用BERT模型进行推断
    with torch.no_grad():
        outputs = model(**encoded_input)

    # 获取BERT模型的输出向量
    output = outputs.last_hidden_state

    return output


# 调用ResNet-34模型对JPG文件进行处理
def process_image(image_path):
    # 加载ResNet-34模型
    model = resnet34(pretrained=True)
    model.eval()

    # 图像预处理
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 读取图像文件
    image = Image.open(image_path)

    # 图像预处理
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)

    # 将输入张量转移到GPU（如果可用）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_batch = input_batch.to(device)

    # 使用ResNet-34模型进行推断
    with torch.no_grad():
        input_batch = input_batch.to(device)
        output = model(input_batch)

    return output


# 将两个输出结果压缩成一个1x512的张量
def combine_outputs(output1, output2):
    tensor_combined = torch.cat((output1.view(1, -1), output2.view(1, -1)), dim=1)
    return tensor_combined


# 调用函数进行处理
json_file = r'G:\CAFE\news_article.json'
image_file = r'G:\CAFE\843039.jpg'

bert_output = process_json(json_file)
resnet_output = process_image(image_file)

combined_output = combine_outputs(bert_output, resnet_output)

combined_output = combined_output.view(512,1)

print(combined_output.shape)

