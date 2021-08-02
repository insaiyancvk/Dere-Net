# Dere-Net
A classifier that classifies persona (dere) of waifus using deep neural netowrks

But what are Dere types? Check this dere chart

![](https://i.pinimg.com/originals/9b/eb/87/9beb870c74adb42c917301563066597b.jpg)

So obviously you must be thinking, how can a neural network classify the persona based on the images? Well it doesn't give proper results. So keeping that in mind, I just used this dataset to learn more about CNNs and Deep learning :)

But the [dataset](https://www.kaggle.com/jahelsantiagoleon/female-anime-characters-anime-dataset) had only 7 Deres, them being:
- Dandere
- Himedere
- Yangire
- Tsundere
- Kuudere
- Deredere
- Yandere

Here are some predictions from [colab](https://colab.research.google.com/github/insaiyancvk/Dere-Net/blob/main/Dere_net_colab.ipynb) (if you watched the anime that these characters belong to, then you'll know the correct dere):

<div>
<img src="./assets/erina.jpeg" width=300px>
<img src="./assets/hancock.jpeg" width=300px>
<img src="./assets/hinata.png" width=300px>
<img src="./assets/mikasa.jpeg" width=300px>
<img src="./assets/rin.png" width=300px>
</div>

TODO:
- Use gradio and make a web UI.
- Host it on heroku.

[TL;DR](https://github.com/insaiyancvk/Dere-Net/blob/main/updates.md)

My Model's best performance:

![](./assets/64bat20epochLR0151.png)

ResNet18's performance on same dataset:

![](./assets/RESNET18.png)
