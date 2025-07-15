@echo off
echo Installing dependencies...
pip install -r requirements.txt

echo Ingesting data into Pinecone...
python ingest_to_pinecone.py

echo Launching Sweet Robo AI Assistant...
streamlit run app.py

pause
