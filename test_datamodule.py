from config import load_config
from data import create_datamodule

# Load config
config = load_config()

# Create datamodule
dm = create_datamodule(config)

# Setup
dm.prepare_data()
dm.setup('fit')

# Get a sample
sample = dm.get_sample('train', idx=0)
print(f"Image shape: {sample['image'].shape}")
print(f"Mask shape: {sample['mask'].shape}")
print(f"Sample ID: {sample['sample_id']}")
print(f"Text: {sample['text_content']}")

# Get dataloaders
train_loader = dm.train_dataloader()
val_loader = dm.val_dataloader()

print(f"\nTrain batches: {len(train_loader)}")
print(f"Val batches: {len(val_loader)}")

# Get one batch
batch = next(iter(train_loader))
print(f"\nBatch images shape: {batch['image'].shape}")
print(f"Batch masks shape: {batch['mask'].shape}")

print("\n Data pipeline works perfectly!")