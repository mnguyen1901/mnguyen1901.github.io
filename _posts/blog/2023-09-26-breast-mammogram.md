---
layout: post
---

# Introduction
The goal of this competition is to identify breast cancer!!!! Can AI read mammogram? The short answer is yes (but the performance is still very poor). Machine learning can be simply broken down into NLP and vision problems. One deals with text and the other, images. For a very long time, different models were developed to solve these problem but there were no robust model that can handle both NLP and vision tasks. It was not until 2021 that a paper came out showing that the most robust NLP model can be used to solve vision problems. So I wanted to test this approach on the (at that time) newly launch [mammogram competition](https://www.kaggle.com/competitions/rsna-breast-cancer-detection).

# Input
![sample_input](images/mammogram_input.png)

## Python
Transform is an important function in preprocessing that can increase the "learning" of a model. It is just like how my attendings can ask me questions about a similar concept in different ways ("What is high blood pressure?" vs "What is hypertension?"). Similarly, by doing this you force the model to look at a breast mammogram in many different angles.
```
data_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(degrees=45, fill=-1)
])
```

Create a TransformedDataset
```
class TransformedDataset(Dataset):
    def __init__(self, X, y, transform):
        self.X = X
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        image = self.X[index]
        cc = image[0]
        mlo = image[1]

        cc = cc.expand(3,-1,-1)
        mlo = mlo.expand(3,-1,-1)

        if self.transform:
            cc = self.transform(cc)
            mlo = self.transform(mlo)
        label = self.y[index]

        return (cc, mlo, label)
```

Here is the output of the transformed data. We can add more transform functions but that will increase the run time!
![transformed](images/mammogram_transformed.png)

For this project, I use the vit 224 model implemented from huggingface.
```
class vit2x(nn.Module):
    def __init__(self, model_name, hidden_dropout_prob, attention_probs_dropout_prob, attention_heads, hidden_layers):
        super(vit2x, self).__init__()

        self.vit = ViTModel.from_pretrained(model_name,
                                            hidden_dropout_prob=hidden_dropout_prob,
                                            attention_probs_dropout_prob=attention_probs_dropout_prob,
                                            num_hidden_layers=hidden_layers,
                                            num_attention_heads=attention_heads)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(768, 1)
        )

    def forward(self, cc, mlo):
        cc_output = self.vit(cc)
        mlo_output = self.vit(mlo)

        cc_output = cc_output.last_hidden_state
        mlo_output = mlo_output.last_hidden_state

        cc_output = cc_output[:, 0, :]
        mlo_output = mlo_output[:, 0, :]

        viewpool = torch.max(torch.stack([cc_output, mlo_output]), 0).values

        outputs = self.classifier(viewpool)

        return nn.Sigmoid()(outputs)
```

