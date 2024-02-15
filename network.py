import torch
from torchvision import models, transforms
import torch.nn as nn
import cv2
from PIL import Image

def initialize_model(num_classes):
        # Initialize these variables which will be set in this if statement. Each of these
        #   variables is model specific.
        model_ft = None
        input_size = 0

        model_ft = models.resnet18(pretrained=True)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

        return model_ft, input_size

def transform_img (img_array, input_size=224): #if img size isnt given, default to 224

        img = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY) #convert to grayscale
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        my_transform = transforms.Compose([transforms.Resize(input_size), transforms.ToTensor()]) #convert to tensor
        my_img = my_transform(img)
        my_img = my_img.unsqueeze(0)
        return my_img
        
def predict(model, img_array):
        tensor = transform_img(img_array)
        outputs = model(tensor)
        _, pred = torch.max(outputs,1)

        return pred.item()


num_classes = 29
PATH = './Resnet.pth'

model, input_size = initialize_model(num_classes)
model.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))
model.eval()

# import torch
# from torchvision import models, transforms
# import torch.nn as nn
# import cv2
# from PIL import Image

# def init_model(num_classes):
#     # Initialize these variables which will be set in this if statement. Each of these
#     #   variables is model specific.
#     model_ft = None
#     input_size = 0

#     model_ft = models.squeezenet1_0(pretrained=True)
#     model_ft.features[0]= nn.Conv2d(3, 96, kernel_size=(5, 5), stride=(2, 2))
#     model_ft.classifier[0] = nn.Dropout(p=0.6, inplace=True)
#     model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
#     model_ft.num_classes = num_classes
#     input_size = 224

#     return model_ft, input_size

# def preprocess_img(img_array, input_size=224):
#     img = cv2.cvtColor(cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY), cv2.COLOR_BGR2RGB)
#     img = Image.fromarray(img)
#     return transforms.Compose([transforms.Resize(input_size), transforms.ToTensor()])(img).unsqueeze(0)

# def get_prediction(model, img_array):
#     outputs = model(preprocess_img(img_array))
#     _, prediction = torch.max(outputs, 1)
#     return prediction.item()

# num_classes = 29
# model_path = './Resnet.pth'

# model, input_size = init_model(num_classes)
# model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
# model.eval()
