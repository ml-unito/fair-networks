import sys
sys.path.append('code')

from bank_marketing_dataset import BankMarketingDataset


ds = BankMarketingDataset()


# majority class accuracy
sums = sum(ds._traindata[2])
print(max(sums) / (sum(sums)))
