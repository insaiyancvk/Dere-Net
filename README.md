# Dere Net

[![](https://img.shields.io/badge/heroku-deployed-green)](https://derenet.herokuapp.com/)

A classifier that classifies persona (dere) of waifus using deep neural netowrks.

Here are some classifications on my waifus from the [app](https://derenet.herokuapp.com)

<div>
<img src="./assets/demo1.png" width=800px>
<img height="20" src="https://upload.wikimedia.org/wikipedia/commons/5/59/Empty.png">
<img src="./assets/demo2.png" width=800px>
</div>
<img height="25" src="https://upload.wikimedia.org/wikipedia/commons/5/59/Empty.png">

---

## But what are Dere types? Check this dere chart
<div>
<img height="20" src="https://upload.wikimedia.org/wikipedia/commons/5/59/Empty.png">
</div>
<img src="https://i.pinimg.com/originals/9b/eb/87/9beb870c74adb42c917301563066597b.jpg" height=500px>

So you must be thinking, how can a neural network classify a persona based on the images? Well it doesn't give exact results, just play around it's fun :). And keeping that in mind, I used this dataset to learn more about CNNs and Deep learning with PyTorch :)

But the [dataset](https://www.kaggle.com/jahelsantiagoleon/female-anime-characters-anime-dataset) had only 7 Deres, them being:
- Dandere
- Himedere
- Yangire
- Tsundere
- Kuudere
- Deredere
- Yandere

[TL;DR](https://github.com/insaiyancvk/Dere-Net/blob/main/updates.md)

My Model's best performance:

![](./assets/64bat20epochLR0151.png)

ResNet18's performance on the dataset:

![](./assets/RESNET18.png)
