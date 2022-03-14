import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import shutil
from pathlib import Path

def cleanLabel(label):
    """Clean format from: "['A']" to "A"."""
    for char in "'[]":
        label = label.replace(char, '')
    return label

def LorR(row):
    if 'right' in row['filename']:
        return 'Right'
    elif 'left' in row['filename']:
        return 'Left'

def copyImages(df, outPath, pathPrefix):
    """Create a folder for each class and place all images of that class into the folder."""
    
    # Find all classes
    classes = df['labels'].unique()
    
    # Build the input path
    df['pathToImage'] = pathPrefix + df['filename']
    
    for char in classes:
        # Filter for the class
        classDf = df[df['labels']==char]
        
        # Build the output directory
        Path(f'{outPath}/{char}').mkdir(parents=True, exist_ok=True)
        
        # Copy each image into the output directory
        for i, row in classDf.iterrows():
            shutil.copy(row['pathToImage'], outPath + char)
            

            
##### BUILD DATASET #####
masterDf = pd.read_csv('fundusClassifier/rawData/ocular_disease_recognition/full_df.csv')
masterDf['labels'] = masterDf['labels'].apply(lambda x: cleanLabel(x))
masterDf['Eye'] = masterDf.apply(lambda row: LorR(row), axis=1)

def plotClassCounts(df):
    sns.set_style('whitegrid')
    sns.set_palette('Set2')
    sns.countplot(data=df, x="labels")
    plt.title('Number of images per disease class')
    plt.xlabel('Disease category')
    plt.ylabel('Image count')
    plt.show()
    
def plotPatients(df):
    sns.set_style('whitegrid')
    sns.set_palette('Set2')
    sns.histplot(data=df, x="Patient Age", y='Patient Sex', 
                 stat='count', binwidth=5, 
                 cbar=True, cbar_kws={'label': 'Image count'})
    plt.title('Sex and age of patients in data set')
    plt.xlabel('Age')
    plt.ylabel('Sex')
    plt.xlim((0, 100))
    plt.show()
    
def plotDiseaseAge(df):
    df = df[df['labels']!='N']
    sns.set_style('whitegrid')
    sns.set_palette('Set2')
    sns.histplot(data=df, x="Patient Age", y='labels', 
                 stat='count', binwidth=5, 
                 cbar=True, cbar_kws={'label': 'Image count'})
    plt.title('Number of images per disease class and age')
    plt.xlabel('Age')
    plt.ylabel('Disease category')
    plt.xlim((0, 100))
    plt.show()
    
plotClassCounts(masterDf)
plotPatients(masterDf)
plotDiseaseAge(masterDf)

