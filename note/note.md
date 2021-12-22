# A Convolutional LSTM based Residual Network for Deepfake Video Detection

From each real and fake video, we extracted 
16 samples such that each sample contains five consecutive frames.
We used multi-task CNN (MTCNN) [46] to detect the face landmark
information inside the extracted frame. Afterward, we used this
landmark information to crop the face from the image and aligned
it to the center. Lastly, we resized all the frames to a 240 × 240
resolution.

连续帧图像作为输入，每段视频取16段sample，每段sample取5帧连续帧。


## ViT结构
```python
VisionTransformer(
  (patch_embed): PatchEmbed(     #输入为(batch_size,c,224,224)
    (proj): Conv2d(3, 768, kernel_size=(16, 16), stride=(16, 16))    #输出为(batch_size,768,14,14)
    (norm): Identity()   #占位，输出等于输入
  )
  (pos_drop): Dropout(p=0.0, inplace=False)
  (blocks): Sequential(
    (12*): Block(
      (norm1): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
      (attn): Attention(
        (qkv): Linear(in_features=768, out_features=2304, bias=True)
        (attn_drop): Dropout(p=0.0, inplace=False)
        (proj): Linear(in_features=768, out_features=768, bias=True)
        (proj_drop): Dropout(p=0.0, inplace=False)
      )
      (drop_path): Identity()
      (norm2): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
      (mlp): Mlp(
        (fc1): Linear(in_features=768, out_features=3072, bias=True)
        (act): GELU()
        (fc2): Linear(in_features=3072, out_features=768, bias=True)
        (drop): Dropout(p=0.0, inplace=False)
      )
    )
  )
  (norm): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
  (pre_logits): Identity()
  (head): Linear(in_features=768, out_features=1000, bias=True)
)
```


timm
ViT
```python
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads      #多头注意力，num_heads为头数,默认为8
        head_dim = dim // num_heads     #每一头的dim数
        self.scale = head_dim ** -0.5   

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape   #B:batch_size, N:squence_length, C:dim
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)    #permute维度变换，permute之前为(B,N,3,8,C/8),之后为(3,B,8,N,C/8)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
        #q,k,v分别(B,8,N,C/8)
        attn = (q @ k.transpose(-2, -1)) * self.scale   #a @ b等同于a.mm(b)或a.matmul(b)   attn(B,8,N,N)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)   #(attn @ v) -> (B,8,N,C/8) ;  transpose -> (B,N,8,C/8) ; reshape -> (B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x   #(B, N, C)
```
