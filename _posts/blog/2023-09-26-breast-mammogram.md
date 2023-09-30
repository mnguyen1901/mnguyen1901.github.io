---
layout: post
---

# Introduction
Can AI accurately interpret mammograms? The concise answer is, yes, although its efficacy remains limited. Traditionally, machine learning has been categorized into NLP (Natural Language Processing) for textual data and vision-related tasks for images. For years, individual models have been developed for each type of task, but none were versatile enough to handle both NLP and vision challenges effectively. However, in 2021, a groundbreaking study demonstrated that an advanced NLP model could also be adapted for vision tasks. Motivated by this discovery, I decided to use this approach in the 2022 [mammogram competition](https://www.kaggle.com/competitions/rsna-breast-cancer-detection). The aim of this competition is to determine the presence of breast cancer through mammograms and the dataset were labeled by radiologists at various institutions.

# Input
![sample_input](images/mammogram_input.png)

## Python
Transform is an important function in preprocessing that can increase the "learning" of a model. It is just like how we can ask questions about a similar concept in different ways ("What is high blood pressure?" vs "What is hypertension?"). Similarly, by doing this you force the model to look at a breast mammogram in many different angles.
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

For this project, I used the vit 224 model implemented by huggingface.
```
model_name_or_path = "google/vit-base-patch16-224-in21k"
vit = vit2x(model_name_or_path, hidden_dropout_prob, attention_probs_dropout_prob, attention_heads, hidden_layers)
```

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


While others prefer to use the training function by huggingface, I prefer to make my own, just so that I have better control of the training process. Interestingly, I found that the initial warmup_steps can help stabilize the "learning" process.

```
epochs = 30
hidden_dropout_prob = 0.2
attention_probs_dropout_prob = 0.2
hidden_layers = 12
attention_heads = 12
lr = 1e-3
```
```
num_training_steps = epochs * len(train_dataloader)
progress_bar = tqdm(range(num_training_steps))

optimizer = AdamW(vit.parameters(), lr=lr, weight_decay=0.01)
lr_scheduler = get_scheduler(name="cosine",
                            optimizer=optimizer,
                            num_warmup_steps=num_training_steps*0.1,
                            num_training_steps=num_training_steps)

best_loss = 1e5
best_f1 = 0
no_improvement = 0
total_loss_train = []
total_loss_val = []
f1_val = []
f1_train = []

vit.train()
for epoch in range(epochs):
    loss_epoch_train = 0

    for batch in train_dataloader:
        cc = batch[0].to(device)
        mlo = batch[1].to(device)
        y = batch[2].to(device)
        outputs = vit(cc, mlo)
        loss = vit.loss_fn(outputs, y)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(vit.parameters(), 1.0)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        progress_bar.update(1)
        loss_epoch_train += loss.item()

        del cc, mlo, y
        torch.cuda.empty_cache()

    total_loss_train.append(loss_epoch_train/len(train_dataloader))
    loss_epoch_val, f1_epoch_val, y_proba, y_true = evaluate(vit, val_dataloader)
    _, f1_epoch_train, _, _ = evaluate(vit, train_dataloader)
    total_loss_val.append(loss_epoch_val)

    f1_val.append(f1_epoch_val)
    f1_train.append(f1_epoch_train)

    print(epoch, " "*(10-len(str(epoch))),
        str(total_loss_train[-1])[:6], " "*9,
        str(loss_epoch_val)[:6], " "*7,
        str(f1_epoch_val)[:4], "/", str(f1_epoch_train)[:4])

    if loss_epoch_val < best_loss:
        best_loss = loss_epoch_val
        no_improvement = 0
        torch.save(vit.state_dict(), "classifier_only.pt")
    elif epoch>20:
        no_improvement += 1

    if no_improvement == 10:
        break
```
