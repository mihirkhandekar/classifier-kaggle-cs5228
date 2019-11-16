python3 -m virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
kaggle competitions download -c 1910-cs5228-knowledge-discovery-and-data-mining
unzip 1910-cs5228-knowledge-discovery-and-data-mining.zip -d data
rm 1910-cs5228-knowledge-discovery-and-data-mining.zip
python src/ml_feat.py
python src/nn.py
python src/ensemble.py