# Amharic Data Gathering

This folder contains a continuous web scraper that keeps only **Amharic** sentences using `langdetect`.

Output file:
- `data_gathering/am.jsonl` (one JSON object per line, one sentence per line)

## Install dependencies

```powershell
python -m pip install requests beautifulsoup4 pyyaml langdetect langid
```

## Run once (smoke test)

```powershell
python data_gathering/amharic_scraper.py --config data_gathering/config.yaml --once --max-pages 10
```

## Run continuously

```powershell
python data_gathering/amharic_scraper.py --config data_gathering/config.yaml
```

Stop with `Ctrl+C`.

## Configure

Edit:
- `data_gathering/config.yaml`
- `data_gathering/seeds.txt`
