# Skills Assessment 1



Your task is to craft a targeted adversarial example using FGSM.
Unlike the simple MNIST challenges, this assessment uses CIFAR-10 with
color images and a more sophisticated classifier. You must transform a
dog image into one the classifier predicts as a cat, while staying
within a strictL∞L_\inftyperturbation budget.





The constraint is‖xadv−x‖∞≤ϵ\lVert x_{\text{adv}} - x \rVert_\infty \leq \epsilonwhereϵ=8/255\epsilon = 8/255in pixel space. All three conditions must hold: the predicted class
equals the target (cat, class 3), the maximum absolute pixel difference
is at mostϵ\epsilon,
and all pixel values remain in[0,1]32×32×3[0,1]^{32 \times 32 \times 3}.



## Quick Start

Start your instance and check readiness:

Code:bash```
`exportBASE_URL="http://instance_ip:port"curl-s"$BASE_URL/health"`
```

## API

The API is small. Each response is JSON. Images are base64`PNG`files of shape`32×32×3`(RGB) in`[0,1]`after decoding.

### GET /health

Service health check.

Code:bash```
`curl-s"$BASE_URL/health"`
```

### GET /challenge

Returns the fixed sample and constraint.

Code:bash```
`curl-s"$BASE_URL/challenge"|jq`
```

Example response shape:

Code:json```
`{"challenge":"I-FGSM Targeted Attack on CIFAR-10","sample_index":42,"original_class":5,"original_class_name":"dog","target_class":3,"target_class_name":"cat","epsilon":0.03137254901960784,"epsilon_description":"0.031373 (8/255)","max_iterations_hint":100,"normalization":{"mean":[0.4914,0.4822,0.4465],"std":[0.247,0.2435,0.2616]},"image":"<base64 PNG>","instructions":"Implement targeted I-FGSM..."}`
```

### GET /model/weights

Download the model weights file (`cifar10_model_best.pth`, ~6.3MB). Required for whitebox gradient computation.

Code:bash```
`curl-s"$BASE_URL/model/weights"-o cifar10_model_best.pth`
```

### POST /predict

Returns the model's prediction for an arbitrary image. Use this to validate your pipeline and intermediate candidates. Does not return the flag.

Code:bash```
`curl-s -X POST"$BASE_URL/predict"\-H'content-type: application/json'\-d'{"image": "<base64 PNG of 32x32 RGB>"}'|jq`
```

Example response shape:

Code:json```
`{"predicted_class":3,"predicted_class_name":"cat","confidence":0.85,"all_probabilities":{"airplane":0.01,"automobile":0.02,...}}`
```

### POST /submit



Validates that your candidate is a proper`32×32×3`RGB
image in`[0,1]`, that theL∞L_\inftydistance to the baseline image is at mostϵ\epsilon,
and that the classifier predicts the target class. Returns the flag on
success.



Code:bash```
`curl-s -X POST"$BASE_URL/submit"\-H'content-type: application/json'\-d'{"image": "<base64 PNG of 32x32 RGB>"}'|jq`
```

Example success response shape:

Code:json```
`{"success":true,"validation":{"linf_norm":0.031373,"linf_satisfied":true,"valid_range":true,"adversarial_class":"cat","target_achieved":true},"flag":"HTB{...}"}`
```

## Model Architecture

The target model is a simple CNN trained on CIFAR-10. Save this code as`model.py`:

Code:python```
`importtorchimporttorch.nnasnnclassCIFAR10CNN(nn.Module):def__init__(self,num_classes:int=10):super(CIFAR10CNN,self).__init__()# First convolutional blockself.conv1=nn.Conv2d(3,32,kernel_size=3,padding=1)self.bn1=nn.BatchNorm2d(32)self.relu1=nn.ReLU()self.pool1=nn.MaxPool2d(2,2)# Second convolutional blockself.conv2=nn.Conv2d(32,64,kernel_size=3,padding=1)self.bn2=nn.BatchNorm2d(64)self.relu2=nn.ReLU()self.pool2=nn.MaxPool2d(2,2)# Fully connected layersself.fc1=nn.Linear(64*8*8,128)self.relu3=nn.ReLU()self.dropout=nn.Dropout(0.5)self.fc2=nn.Linear(128,num_classes)defforward(self,x:torch.Tensor)->torch.Tensor:x=self.pool1(self.relu1(self.bn1(self.conv1(x))))x=self.pool2(self.relu2(self.bn2(self.conv2(x))))x=x.view(x.size(0),-1)# Flattenx=self.dropout(self.relu3(self.fc1(x)))x=self.fc2(x)returnxdefload_model(model_path:str,device:str="cuda")->CIFAR10CNN:model=CIFAR10CNN(num_classes=10)# Load checkpointcheckpoint=torch.load(model_path,map_location=device)# Handle both direct state_dict and checkpoint dict formatsifisinstance(checkpoint,dict)and"model_state_dict"incheckpoint:model.load_state_dict(checkpoint["model_state_dict"])else:model.load_state_dict(checkpoint)model=model.to(device)model.eval()returnmodel# CIFAR-10 class namesCIFAR10_CLASSES=["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck",]# Normalization parameters (computed from CIFAR-10 training set)NORMALIZATION_MEAN=[0.4914,0.4822,0.4465]NORMALIZATION_STD=[0.247,0.2435,0.2616]`
```

## Minimal Python Scaffolds

The following helpers fetch the challenge, convert between base64`PNG`and`[0,1]`tensors, and call the API.

Code:python```
`importos,io,base64,requestsimporttorchimporttorch.nn.functionalasFimporttorchvision.transformsastransformsfromPILimportImageimportnumpyasnp

BASE_URL=os.getenv("BASE_URL","http://127.0.0.1:8000")deftensor_from_b64_png(b64:str)->torch.Tensor:raw=base64.b64decode(b64)img=Image.open(io.BytesIO(raw))ifimg.size!=(32,32):raiseValueError("Expected 32x32 PNG")tensor=transforms.ToTensor()(img)# converts to (3, 32, 32) in [0,1]returntensordefb64_png_from_tensor(tensor:torch.Tensor)->str:img_array=(tensor.permute(1,2,0).numpy()*255).astype(np.uint8)img=Image.fromarray(img_array)buf=io.BytesIO()img.save(buf,format="PNG")buf.seek(0)returnbase64.b64encode(buf.getvalue()).decode("ascii")deflinf(a:torch.Tensor,b:torch.Tensor)->float:returnfloat(torch.abs(a-b).max())# Download model weightsweights_path="cifar10_model_best.pth"ifnotos.path.exists(weights_path):print("Downloading model weights...")resp=requests.get(f"{BASE_URL}/model/weights")withopen(weights_path,"wb")asf:f.write(resp.content)print(f"Saved to{weights_path}")# Load model (assumes model.py from architecture section above)frommodelimportload_model,NORMALIZATION_MEAN,NORMALIZATION_STD

device="cuda"iftorch.cuda.is_available()else"cpu"model=load_model(weights_path,device=device)print(f"Model loaded on{device}")# Fetch challengech=requests.get(f"{BASE_URL}/challenge",timeout=10).json()x=tensor_from_b64_png(ch["image"])# (3, 32, 32)orig_class=int(ch["original_class"])# 5 (dog)target_class=int(ch["target_class"])# 3 (cat)epsilon=float(ch["epsilon"])# 8/255mean=torch.tensor(ch["normalization"]["mean"]).view(3,1,1)std=torch.tensor(ch["normalization"]["std"]).view(3,1,1)# Verify clean predictionx_norm=(x-mean)/stdwithtorch.no_grad():pred=model(x_norm.unsqueeze(0).to(device)).argmax(dim=1).item()print(f"Original: class{orig_class}, Target: class{target_class}, Clean pred:{pred}, Epsilon:{epsilon:.6f}")`
```

To verify server-side checks, submit the clean image and observe expected failures:

Code:python```
`bad=requests.post(f"{BASE_URL}/submit",json={"image":b64_png_from_tensor(x)},timeout=10)print(bad.status_code,bad.text)# expected: success=false, "Target not achieved"`
```