## Time Series Data Augmentation for Deep Learning: A Survey

>WEN, Qingsong, et al. Time series data augmentation for deep learning: A survey. 
>
>*arXiv preprint arXiv:2002.12478*, 2020.

[[2002.12478\] Time Series Data Augmentation for Deep Learning: A Survey (arxiv.org)](https://arxiv.org/abs/2002.12478)

### Basic Approaches

| Domain               | Method                   |                                                              |
| -------------------- | ------------------------ | ------------------------------------------------------------ |
| **Time Domain**      | 1. Direct Transformation | - 원본 데이터에 직접적으로 변형을 가하는 방법. <br />- 연구자의 임의로 적당하게 데이터를 수정<br />- Noise Injection도 여기에 해당 |
|                      | 2. Slicing window        | slicing window<br />cropping image<br />원본 데이터에서 연속적인 윈도우를 생성해 잘라내며 윈도우의 길이를 임의로 조절. (이런 방식으로 하나의 데이터셋을 여러개 생성할 수 있음.) |
|                      | 3. Window Warping        | - DTW와 유사<br />- 임의의 구간을 **압축하거나 확장**.<br />- 시퀀스 전체 길이에 영향을 준다. |
|                      | 4. Flipping              | - 데이터의 부호를 변환. 양수는 음수로 음수는 양수로          |
|                      | 5. Ensemble              | - DTW를 실행하고 각 window sample들에 가중치가 곱해진 ensemble version의 샘플을 활용 |
|                      | 6. Noise Injection       | - 작은 크기의 noise나 outlier를 삽입.<br />(1) <u>Gaussian noise</u><br />(2) <u>Spike</u> : 임의의 인덱스를 택해 임의의 값을 대입.<br />스파이크성 잡음<br />![image-20220708071936601](.\imgs\image-20220708071936601.png)<br /><br />![image-20220708074832462](.\imgs\image-20220708074832462.png)<br />(3) <u>Step-like trend</u> : left index에서부터 right index까지의 spike 누적합<br />(4) <u>slope-like trend</u> : 기존 데이터셋에 선형 값을 대입 |
|                      | 7. Label Expansion       |                                                              |
| **Frequency Domain** |                          |                                                              |

