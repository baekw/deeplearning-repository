# GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium (**2017**)

FID는 Network의 중간 layer에서 feature를 가져와 이를 활용한다.

실제 데이터의 분포를 활용하지 않는 단점을 보완해 실제 데이터와 생성 데이터에서 얻은 feature들의 평균과 공분산을 비교하는 식으로 구성된다. 

실제 데이터와 생성된 데이터의 분포를 가우시안이라고 가정한다. 

$FID(x,g)=||\mu_X-\mu_g||^2+Tr(\Sigma_x+\Sigma_g-2(\Sigma_x\Sigma_g)^2)$



수학적으로, FID 거리는 두 "다변량" 정규분포 사이의 거리를 계산하는데 사용됨. "일변량" 정규분포의 경우, 프레쳇 거리는 다음과 같이 계산됨.

$d(X,Y)=(\mu_X+\mu_Y)^2+(\sigma_X-\sigma_Y)^2$

여기서 $\mu$와 $\sigma$는 정규분포의 평균 및 표준편차이며, $X$와 $Y$는 두 개의 정규분포이다.



### FID를 사용하여 GAN 평가하기

1. FID는 Frechet Inception Distance이며, 각 이미지를 요약하기 위한 Inception V3모델에서 활성화를 사용하면 스코어에 FID라는 이름이 부여됨.

2. 실제이미지와 생성이미지에 대한 임베딩을 계산한다. 본 논문의 저자는 최소 샘플 사이즈 10,000를 사용하여 FID를 계산할 것을 권장한다. 그렇지 않은 경우 generator의 실제 FID값이 과소평가된다.

   ```python
   def compute_embeddings(dataloader, count):
       image_embeddings = []
       
       for _ in tqdm(range(count)):
           images = next(iter(dataloader))
           embeddings = inception_model.predict(images)
           
           image_embeddings.extend(embeddings)
       return np.array(image_embeddings)
   
   count = math.ceil(10000/BATCH_SIZE)
   
   # compute embeddings for real images
   real_image_embeddings = compute_embeddings(trainloader, count)
   
   # compute embeddings for generated images
   generated_image_embeddings = compute_embeddings(genloader, count)
   
   real_image_embeddings.shape, generated_image_embeddings.shape
   ```

   

3. 실제 생성된 이미지 임베딩을 이용하여 FID 스코어를 계산한다.

   ```python
   def calculate_fid(real_embeddings, generated_embeddings):
       #calculate mean and covariance statistics
       mu1, sigma1 = real_embeddings.mean(axis=0), np.conv(real_embeddings, rowvar = False)
       mu2, sigma2 = generated_embeddings.mean(axis=0), np.conv(real_embeddings, rowvar = False)
       
       #calculate sum squared difference between means
       ssidff = np.sum((mu1-mu2)**2)
       #calculate sqrt of product between cov
       convmean = linalg.sqrtm(sigma1.dot(sigma2))
       #check and correct imaginary numbers from sqrt
       if np.iscomplexobj(convmean):
           convmean = convmean.real
       #calculate score
       fid = ssdiff + np.trace(sigma1 + sigma2 - 2*convmean)
       return fid
   
   fid = calculate_fid(real_image_embeddings, generated_image_embeddings)
   ```







>출처
>
>https://wandb.ai/wandb_fc/korean/reports/-Frechet-Inception-distance-FID-GANs---Vmlldzo0MzQ3Mzc

