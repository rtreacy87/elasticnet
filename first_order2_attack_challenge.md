# Skills Assessment 2



Your task is to implement the DeepFool algorithm to find a minimalL2L_2perturbation that causes misclassification on CIFAR-10. This is an
untargeted attack where any misclassification succeeds.





The constraint is‖xadv−xσ‖2≤threshold\lVert \frac{x_{\text{adv}} - x}{\sigma} \rVert_2 \leq \text{threshold}where theℓ2\ell_2norm is measured in normalized space. All three conditions must hold:
the predicted class differs from the original (any class except horse),
theℓ2\ell_2distance in normalized space is at most the threshold (3.5), and all
pixel values remain in[0,1]32×32×3[0,1]^{32 \times 32 \times 3}.



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
`{"challenge":"DeepFool Minimal Perturbation on CIFAR-10","sample_index":137,"original_class":7,"original_class_name":"horse","l2_threshold":3.5,"num_classes_hint":10,"overshoot_hint":0.02,"max_iterations_hint":50,"normalization":{"mean":[0.4914,0.4822,0.4465],"std":[0.247,0.2435,0.2616]},"image":"<base64 PNG>","instructions":"Find minimal L2 perturbation..."}`
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
`{"predicted_class":9,"predicted_class_name":"truck","confidence":0.62,"all_probabilities":{"airplane":0.01,"automobile":0.02,...}}`
```

### POST /submit



Validates that your candidate is a proper`32×32×3`RGB
image in`[0,1]`, that theℓ2\ell_2distance to the baseline image in normalized space is at most`l2_threshold`, and that the classifier misclassifies it.
Returns the flag on success.



Code:bash```
`curl-s -X POST"$BASE_URL/submit"\-H'content-type: application/json'\-d'{"image": "<base64 PNG of 32x32 RGB>"}'|jq`
```

Example success response shape:

Code:json```
`{"success":true,"validation":{"l2_norm":0.9632,"l2_threshold":3.5,"l2_satisfied":true,"valid_range":true,"original_class":"horse","adversarial_class":"truck","misclassification":true},"flag":"HTB{...}"}`
```

## Model Architecture

The target model is a simple CNN trained on CIFAR-10. Save this code as`model.py`:

Code:python```
`importtorchimporttorch.nnasnnclassCIFAR10CNN(nn.Module):def__init__(self,num_classes:int=10):super(CIFAR10CNN,self).__init__()# First convolutional blockself.conv1=nn.Conv2d(3,32,kernel_size=3,padding=1)self.bn1=nn.BatchNorm2d(32)self.relu1=nn.ReLU()self.pool1=nn.MaxPool2d(2,2)# Second convolutional blockself.conv2=nn.Conv2d(32,64,kernel_size=3,padding=1)self.bn2=nn.BatchNorm2d(64)self.relu2=nn.ReLU()self.pool2=nn.MaxPool2d(2,2)# Fully connected layersself.fc1=nn.Linear(64*8*8,128)self.relu3=nn.ReLU()self.dropout=nn.Dropout(0.5)self.fc2=nn.Linear(128,num_classes)defforward(self,x:torch.Tensor)->torch.Tensor:x=self.pool1(self.relu1(self.bn1(self.conv1(x))))x=self.pool2(self.relu2(self.bn2(self.conv2(x))))x=x.view(x.size(0),-1)# Flattenx=self.dropout(self.relu3(self.fc1(x)))x=self.fc2(x)returnxdefload_model(model_path:str,device:str='cuda')->CIFAR10CNN:model=CIFAR10CNN(num_classes=10)# Load checkpointcheckpoint=torch.load(model_path,map_location=device)# Handle both direct state_dict and checkpoint dict formatsifisinstance(checkpoint,dict)and'model_state_dict'incheckpoint:model.load_state_dict(checkpoint['model_state_dict'])else:model.load_state_dict(checkpoint)model=model.to(device)model.eval()returnmodel# CIFAR-10 class namesCIFAR10_CLASSES=['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']# Normalization parameters (computed from CIFAR-10 training set)NORMALIZATION_MEAN=[0.4914,0.4822,0.4465]NORMALIZATION_STD=[0.247,0.2435,0.2616]`
```

## Minimal Python Scaffolds

The following helpers fetch the challenge, convert between base64`PNG`and`[0,1]`tensors, and call the API.

Code:python```
`importos,io,base64,requestsimporttorchimporttorch.nn.functionalasFimporttorchvision.transformsastransformsfromPILimportImageimportnumpyasnp

BASE_URL=os.getenv("BASE_URL","http://127.0.0.1:8001")deftensor_from_b64_png(b64:str)->torch.Tensor:raw=base64.b64decode(b64)img=Image.open(io.BytesIO(raw))ifimg.size!=(32,32):raiseValueError("Expected 32x32 PNG")tensor=transforms.ToTensor()(img)# converts to (3, 32, 32) in [0,1]returntensordefb64_png_from_tensor(tensor:torch.Tensor)->str:img_array=(tensor.permute(1,2,0).numpy()*255).astype(np.uint8)img=Image.fromarray(img_array)buf=io.BytesIO()img.save(buf,format="PNG")buf.seek(0)returnbase64.b64encode(buf.getvalue()).decode("ascii")defl2_normalized_space(a:torch.Tensor,b:torch.Tensor,mean,std)->float:mean_t=torch.tensor(mean).view(3,1,1)std_t=torch.tensor(std).view(3,1,1)a_norm=(a-mean_t)/std_t
    b_norm=(b-mean_t)/std_treturnfloat(torch.norm(a_norm-b_norm))# Download model weightsweights_path="cifar10_model_best.pth"ifnotos.path.exists(weights_path):print("Downloading model weights...")resp=requests.get(f"{BASE_URL}/model/weights")withopen(weights_path,"wb")asf:f.write(resp.content)print(f"Saved to{weights_path}")# Load model (assumes model.py from architecture section above)frommodelimportload_model,NORMALIZATION_MEAN,NORMALIZATION_STD

device="cuda"iftorch.cuda.is_available()else"cpu"model=load_model(weights_path,device=device)print(f"Model loaded on{device}")# Fetch challengech=requests.get(f"{BASE_URL}/challenge",timeout=10).json()x=tensor_from_b64_png(ch["image"])# (3, 32, 32)orig_class=int(ch["original_class"])# 7 (horse)l2_threshold=float(ch["l2_threshold"])# 3.5mean=ch["normalization"]["mean"]std=ch["normalization"]["std"]mean_t=torch.tensor(mean).view(3,1,1)std_t=torch.tensor(std).view(3,1,1)# Verify clean predictionx_norm=(x-mean_t)/std_twithtorch.no_grad():pred=model(x_norm.unsqueeze(0).to(device)).argmax(dim=1).item()print(f"Original: class{orig_class}, Clean pred:{pred}, L2 threshold:{l2_threshold}")`
```

To verify server-side checks, submit the clean image and observe expected failures:

Code:python```
`bad=requests.post(f"{BASE_URL}/submit",json={"image":b64_png_from_tensor(x)},timeout=10)print(bad.status_code,bad.text)# expected: success=false, "Misclassification not achieved"`
```