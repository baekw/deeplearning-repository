# Denoising Diffusion Probabilistic Models



## 2. Generative Model Overview

### Diffusion based generative model

![image-20221004235929063](D:\Onedrive\OneDrive - 인하대학교\papers-repository\review\diffusion_model\img\image-20221004235929063.png)

- Iterative Markov chain 사용하며, 생성에 활용되는 조건부 확률분포 $P_{\theta}(x_{t-1}|x_t)$를 학습시키기 위해 Diffusion Process $q(x_t|x_{t-1})$를 활용

- 타 모델과의 두 가지의 차이점이 존재

  (1) Gaussian Noise를 주입하는 forward과정을 학습 대상으로 생각하지 않음.

  (2) 조건부 확률분포의 체인(Markov chain)으로 이루어짐



## 3. Diffusion model

- 패턴 생성과정을 학습하기 위해 고의적으로 패턴을 무너트리고(Diffusion process), 이를 **다시 복원(Reverse process, Denoising process)하는 조건부 PDF를 학습**

- $q(X_t|X_{t-1})$을 안다고 해도 $q(X_{t-1}|X_t)$을 바로 알 수는 없음. 따라서 학습을 해야함. 다만 $q(X_t|X_{t-1})$가 Gaussian 분포를 따르면 그 역도 Gaussian 분포를 따른다.
- $p_{\theta}(X_{t-1}|X_t)$를 학습해서 이 값이 $q(X_{t-1}|X_t)$에 근접할 수 있도록 학습한다.
- large number of small perturbation. 큰 변화를 매우 작은 양으로 만듦. 1000번의 step으로 잘개 쪼갬.

### 3.1 Forward process

![image-20221005001005893](D:\Onedrive\OneDrive - 인하대학교\papers-repository\review\diffusion_model\img\image-20221005001005893.png)

- 노이즈가 주입되는 양을 $\beta_t$로 표현한다.

- $\beta _t$는 점진적으로 커짐.

```python
n_times = 1000, beta_minmax=[1e-4, 2e-2]

beta_1, beta_2 = beta_minmax
betas = torch.linspace(start = beta_1, end = beta_2, steps = n_times).to(device) # follows DDPM paper
self.sqrt_betas = torch.sqrt(betas)
```

- diffusion noising 과정이 1000번으로 나뉘어서 주입됨.

```python
def make_noisy(self, x_zeros, t):
    epsilon = torch.randn_like(x_zeros).to(self.device)

    sqrt_alpha_bar = self.extract(self.sqrt_alpha_bars, t, x_zeros.shape)
    sqrt_one_minus_alpha_bar = self.extract(self.sqrt_one_minus_alpha_bars, t, x_zeros.shape)

    # make noisy sample. forward process with fixed variance schedule
    noisy_sample = x_zeros * sqrt_alpha_bar + epsilon*sqrt_one_minus_alpha_bar

    return noisy_sample.detach(), epsilon # noise detach는 텐서를 복사하는 방법.
```

- $q(X_t|X_{t-1}) = \sqrt {1-\beta_t} X_{t-1} + \sqrt {\beta_t} \epsilon_{t-1}$





















