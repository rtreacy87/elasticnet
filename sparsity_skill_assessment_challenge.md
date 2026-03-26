# Skills Assessment

Apply your knowledge of sparsity attacks to craft a targeted adversarial example against a ResNet-18 classifier trained on CIFAR-10. You must submit a base64-encoded PNG of shape`32x32x3`that achieves targeted misclassification using either EAD or JSMA. The evaluator validates method signatures and enforces anti-cheating measures including minimum perturbation thresholds.

## Quick start

Begin by verifying the evaluator is ready, then fetch the challenge specification to obtain your sample image, its baseline prediction, and the target class.

```
icantthinkofaname23@htb[/htb]`$exportBASE_URL="http://instance_ip:port"`
```

```
icantthinkofaname23@htb[/htb]`$curl-s"$BASE_URL/health"|jq{
  "status": "ok",
  "model": "...",
  "items": ...
}`
```

```
icantthinkofaname23@htb[/htb]`$curl-s"$BASE_URL/challenge"|jq{
  "items": [
    {
      "sample_id": ...,
      "label": ...,
      "target": ...,
      "required_method": "...",
      "image_b64": "..."
    }
  ]
}`
```

Next, download the model metadata and weights. The architecture is ResNet-18 adapted for CIFAR-10, with the full implementation provided in the scaffolds below.

```
icantthinkofaname23@htb[/htb]`$curl-s"$BASE_URL/model"|jq{
  "arch": "ResNetCIFAR",
  "weights_sha256": "...",
  "weights_size": ...,
  "normalize": {
    "mean": [
      ...
    ],
    "std": [
      ...
    ]
  },
  "weights_url": "/model/weights"
}`
```

```
icantthinkofaname23@htb[/htb]`$curl-sL"$BASE_URL/model/weights"-o cifar10_model.pth`
```

During development, use the`/predict`endpoint to validate your image encoding and PNG round-trip behavior before final submission.

```
icantthinkofaname23@htb[/htb]`$curl-s -X POST"$BASE_URL/predict"\
  -H 'content-type: application/json' \
  -d '{"image_b64": "<base64 PNG of 32x32 RGB>"}' | jq`
```

## Minimal Python scaffolds

The helpers below provide image I/O and HTTP calls. They are for integration only.

Code:python```
`importio,os,base64,json,urllib.requestfromtypingimportDict,AnyfromPILimportImageimportnumpyasnp

BASE_URL=os.getenv("BASE_URL","http://127.0.0.1:8000")defb64_from_x01(x4d:np.ndarray)->str:x=np.transpose(x4d[0],(1,2,0))x255=np.clip((x*255.0).round(),0,255).astype(np.uint8)img=Image.fromarray(x255,mode="RGB")buf=io.BytesIO()img.save(buf,format="PNG",optimize=True)returnbase64.b64encode(buf.getvalue()).decode("ascii")defx01_from_b64(b64:str)->np.ndarray:raw=base64.b64decode(b64)img=Image.open(io.BytesIO(raw)).convert("RGB")x=np.asarray(img,dtype=np.float32)/255.0returnnp.transpose(x,(2,0,1))[None,...].astype(np.float32)defhttp_get(path:str)->Dict[str,Any]:withurllib.request.urlopen(f"{BASE_URL}{path}",timeout=15)asr:returnjson.loads(r.read().decode("utf-8"))defhttp_post(path:str,body:Dict[str,Any])->Dict[str,Any]:data=json.dumps(body).encode("utf-8")req=urllib.request.Request(f"{BASE_URL}{path}",data=data,headers={"Content-Type":"application/json"},method="POST",)withurllib.request.urlopen(req,timeout=30)asr:returnjson.loads(r.read().decode("utf-8"))ch=http_get("/challenge")meta=http_get("/model")print({"items":len(ch["items"]),"arch":meta["arch"],"weights_url":meta["weights_url"],})`
```

Load the model locally with`ResNetCIFAR`and the downloaded weights.

Code:python```
`importtorch,torch.nnasnnclassBasicBlock(nn.Module):expansion=1def__init__(self,in_planes,planes,stride=1):super().__init__()self.conv1=nn.Conv2d(in_planes,planes,3,stride=stride,padding=1,bias=False)self.bn1=nn.BatchNorm2d(planes)self.conv2=nn.Conv2d(planes,planes,3,padding=1,bias=False)self.bn2=nn.BatchNorm2d(planes)self.shortcut=nn.Sequential()ifstride!=1orin_planes!=planes:self.shortcut=nn.Sequential(nn.Conv2d(in_planes,planes,1,stride=stride,bias=False),nn.BatchNorm2d(planes),)defforward(self,x):out=torch.relu(self.bn1(self.conv1(x)))out=self.bn2(self.conv2(out))out+=self.shortcut(x)returntorch.relu(out)classResNetCIFAR(nn.Module):def__init__(self,num_blocks=(2,2,2,2),num_classes=10):super().__init__()self.in_planes=64self.conv1=nn.Conv2d(3,64,3,1,1,bias=False)self.bn1=nn.BatchNorm2d(64)self.layer1=self._make_layer(64,num_blocks[0],1)self.layer2=self._make_layer(128,num_blocks[1],2)self.layer3=self._make_layer(256,num_blocks[2],2)self.layer4=self._make_layer(512,num_blocks[3],2)self.avgpool=nn.AdaptiveAvgPool2d(1)self.fc=nn.Linear(512,num_classes)def_make_layer(self,planes,n,stride):layers=[]forsin[stride]+[1]*(n-1):layers.append(BasicBlock(self.in_planes,planes,s))self.in_planes=planesreturnnn.Sequential(*layers)defforward(self,x):out=torch.relu(self.bn1(self.conv1(x)))out=self.layer1(out)out=self.layer2(out)out=self.layer3(out)out=self.layer4(out)out=self.avgpool(out)out=torch.flatten(out,1)returnself.fc(out)defcifar_normalize(x):mean=torch.tensor((0.4914,0.4822,0.4465),dtype=x.dtype,device=x.device)[None,:,None,None]std=torch.tensor((0.2470,0.2435,0.2616),dtype=x.dtype,device=x.device)[None,:,None,None]return(x-mean)/std


ckpt_path="cifar10_model.pth"urllib.request.urlretrieve(f"{BASE_URL}{meta['weights_url']}",ckpt_path)device=torch.device("cuda"iftorch.cuda.is_available()else"cpu")model=ResNetCIFAR().to(device).eval()state=torch.load(ckpt_path,map_location=device)state_dict=state.get("state_dict_ema")orstate.get("state_dict")orstate
model.load_state_dict(state_dict)`
```

With the model loaded, craft either an EAD or JSMA attack to generate your adversarial example. Before submission, perform a PNG round-trip on your candidate image to mirror the evaluator's decode path and ensure your perturbation survives the encoding process.

## Submission

Submit your adversarial example to the`POST /submit_images`endpoint with the correct`sample_id`and method tag. Images are transmitted as base64-encoded PNGs and automatically decoded to the`[0,1]`range on the server for evaluation.

Code:json```
`POST /submit_images{"items":[{"sample_id":0,"method":"ead","image_b64":"..."}]}`
```

The evaluator enforces a minimum L2 perturbation threshold of 1.5 to prevent submitting unmodified clean images. This anti-cheating measure ensures your adversarial example contains actual perturbations rather than the original baseline image. When all validation checks pass (targeted misclassification, method signature, perturbation threshold), the response includes your flag.