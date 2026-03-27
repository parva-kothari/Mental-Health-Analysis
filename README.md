## Requirements:
- Python: >=3.8 and <=3.12
- requirements.txt
- cuda (requirement for practical needs of training)

### How to download dependencies from requirements.txt
```bash
py -3.11 -m pip install -r requirements.txt
```
---

## Troubleshooting (Issues to be resolved later)

- If you get error because there is no class_weights or label_mapping, search for them in models/mental_health_roberta and copy paste in required models folder
- If you are not able to run hybrid model, copy paste the config.json from data/ to required hybrid model directory


## How to Setup Gemini_Api_Key

1. Go to Search Menu of Windows
2. Search and Navigate for "Environment Variables"
3. In the User Variables Section, Create a new Variable with name "GEMINI_API_KEY" and value as your key.
4. Save and exit

## How to Run Streamlit

```bash
streamlit run app.py
```
or
```bash
py -3.11 -m streamlit run app.py
```