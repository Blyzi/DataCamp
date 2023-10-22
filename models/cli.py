import torch
import lightning.pytorch as pl
import pandas as pd
import sys
from classifier import DataModule, MultiLabelImageClassifierModel, LModule, MultiLabelDataset, transform_image, normalize_image, IMAGE_SIZE, BATCH_SIZE

MODEL_NAME = 'Hope'
LABEL_RARITY_THRESHOLD = 20
TEST_SCORE_THRESHOLD = 0.4

def train(train_dataset, val_dataset, test_dataset, one_hot_labels):
    data_loader = DataModule(train_dataset, val_dataset, test_dataset, batch_size=BATCH_SIZE)

    model = MultiLabelImageClassifierModel(num_classes=len(one_hot_labels), input_size=IMAGE_SIZE, num_channels=7)

    lmodule = LModule(model, one_hot_labels, lr=1e-3, epochs=10, data_loader=data_loader.train_dataloader())

    # Train the model
    print('Starting training.')
    trainer = pl.Trainer(max_epochs=10)
    trainer.fit(lmodule, data_loader.train_dataloader(), data_loader.val_dataloader())
    print('Training finished.')

    # Save model
    torch.save(model.state_dict(), f'./{MODEL_NAME}.pth')

    print('Done.')


def test(index, test_dataset, one_hot_labels, label_names):
    model = MultiLabelImageClassifierModel(num_classes=len(one_hot_labels), input_size=IMAGE_SIZE, num_channels=7)
    model.load_state_dict(torch.load(f'./{MODEL_NAME}.pth'))

    model.eval()

    image, label = test_dataset[index]

    image = image.unsqueeze(0)
    pred = model(image)

    print(f'Threshold: {TEST_SCORE_THRESHOLD} ({int(TEST_SCORE_THRESHOLD * 100)}%)')
    print('Image: ' + test_dataset.df.iloc[index]['file_name'])
    print('Labels: ' + ', '.join([label_names[i] for i, x in enumerate(label) if x == 1]))
    print('Predictions: ' + ', '.join([label_names[i] for i, x in enumerate(pred[0]) if x > TEST_SCORE_THRESHOLD]))
    print('')

    predictions = sorted(enumerate(pred[0]), key=lambda x: x[1], reverse=True)

    for label_idx, score in predictions:
        # print(label_idx, score)
        guessed = score > TEST_SCORE_THRESHOLD
        correct = guessed == label[label_idx]

        if correct:
            print(f"✅   Correct for {label_names[label_idx]}: {float(score) * 100:.2f}% (Guessed: {'Present' if guessed else 'Absent'})")
        else:
            print(f"❌ Incorrect for {label_names[label_idx]}: {float(score) * 100:.2f}% (Guessed: {'Present' if guessed else 'Absent'}, Expected: {'Present' if label[label_idx] else 'Absent'})")


if __name__ == '__main__':
    # Get first CLI arg
    mode = sys.argv[1]

    if mode not in ['train', 'test']:
        print('Invalid mode. Use either "train" or "test".')
        sys.exit()

    data_dir = '../datasets/training/'
    val_dir = '../datasets/evaluation/'
    test_dir = '../datasets/test/'

    train_df = pd.read_csv(data_dir + 'metadata.csv')
    val_df = pd.read_csv(val_dir + 'metadata.csv')
    test_df = pd.read_csv(test_dir + 'metadata.csv')

    train_df['file_name'] = train_df['file_name'].apply(lambda x: data_dir + x)
    val_df['file_name'] = val_df['file_name'].apply(lambda x: val_dir + x)
    test_df['file_name'] = test_df['file_name'].apply(lambda x: test_dir + x)

    one_hot_labels = train_df.drop('file_name', axis=1).sum()[train_df.drop('file_name', axis=1).sum() > LABEL_RARITY_THRESHOLD].index
    label_names = train_df.drop('file_name', axis=1).columns

    train_dataset = MultiLabelDataset(train_df, transforms=transform_image, label_list=one_hot_labels, normalize=normalize_image)
    val_dataset = MultiLabelDataset(val_df, transforms=transform_image, label_list=one_hot_labels, normalize=normalize_image)
    test_dataset = MultiLabelDataset(test_df, transforms=transform_image, label_list=one_hot_labels, normalize=normalize_image)

    print('Setup finished')

    if mode == 'train':
        train(train_dataset, val_dataset, test_dataset, one_hot_labels)
    elif mode == 'test':
        index = int(sys.argv[2] if len(sys.argv) > 2 else 0)
        test(index, test_dataset, one_hot_labels, label_names)
