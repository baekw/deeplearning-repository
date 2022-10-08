# MNIST-C: A Robustness Benchmark for Computer Vision





| MNIST-C       | Noise Category                                               | Figure                                                       |
| ------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| brightness    |                                                              | ![image-20221005212243980](D:\Onedrive\OneDrive - 인하대학교\papers-repository\review\diffusion_model\img\image-20221005212243980.png) |
| canny_edges   |                                                              | ![image-20221005212251714](D:\Onedrive\OneDrive - 인하대학교\papers-repository\review\diffusion_model\img\image-20221005212251714.png) |
| dotted_line   |                                                              | ![image-20221005212258117](D:\Onedrive\OneDrive - 인하대학교\papers-repository\review\diffusion_model\img\image-20221005212258117.png) |
| fog           |                                                              | ![image-20221005212304135](D:\Onedrive\OneDrive - 인하대학교\papers-repository\review\diffusion_model\img\image-20221005212304135.png) |
| glass_blur    | x = np.uint8(gaussian(np.array(x) / 255., sigma=c[0], multichannel=True) * 255) | ![image-20221005212311363](D:\Onedrive\OneDrive - 인하대학교\papers-repository\review\diffusion_model\img\image-20221005212311363.png) |
| impulse_noise | x = sk.util.**random_noise**(np.array(x) / 255., mode='s&p', amount=c) | x = np.clip(x, 0, 1) * 255![image-20221005212321496](D:\Onedrive\OneDrive - 인하대학교\papers-repository\review\diffusion_model\img\image-20221005212321496.png) |
| motion_blur   | x.motion_blur(radius=c[0], sigma=c[1], angle=np.random.uniform(-45, 45)) | ![image-20221005212328970](D:\Onedrive\OneDrive - 인하대학교\papers-repository\review\diffusion_model\img\image-20221005212328970.png) |
| rotate        |                                                              | ![image-20221005212335221](D:\Onedrive\OneDrive - 인하대학교\papers-repository\review\diffusion_model\img\image-20221005212335221.png) |
| scale         |                                                              | ![image-20221005212342334](D:\Onedrive\OneDrive - 인하대학교\papers-repository\review\diffusion_model\img\image-20221005212342334.png) |
| shear         |                                                              | ![image-20221005212349375](D:\Onedrive\OneDrive - 인하대학교\papers-repository\review\diffusion_model\img\image-20221005212349375.png) |
| shot_noise    | x = np.clip(np.random.**poisson**(x * c) / float(c), 0, 1) * 255 | ![image-20221005212355958](D:\Onedrive\OneDrive - 인하대학교\papers-repository\review\diffusion_model\img\image-20221005212355958.png) |
| spatter       |                                                              | ![image-20221005212402532](D:\Onedrive\OneDrive - 인하대학교\papers-repository\review\diffusion_model\img\image-20221005212402532.png) |
| stripe        |                                                              | ![image-20221005212408903](D:\Onedrive\OneDrive - 인하대학교\papers-repository\review\diffusion_model\img\image-20221005212408903.png) |
| translate     |                                                              | ![image-20221005212417172](D:\Onedrive\OneDrive - 인하대학교\papers-repository\review\diffusion_model\img\image-20221005212417172.png) |
| zigzag        |                                                              | ![image-20221005212422853](D:\Onedrive\OneDrive - 인하대학교\papers-repository\review\diffusion_model\img\image-20221005212422853.png) |

