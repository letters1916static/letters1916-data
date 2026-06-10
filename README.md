# letters1916-data

Repo holding TEI/XML data for Letters 1916–1923

## folder struct

### data/editions

holds the TEI/XML files for the letters

### data/indices

holds TEI/XML index files for persons (`listperson.xml`) and places (`listplace.xml`)

### data/meta

holds TEI/XML encoded meta texts, e.g. about the project (`about.xml`)


## llm processing

* add openAI credentials to `secret.env` (see `default.env`)
* set them as env-varibles

```bash
source set_env_varibales.sh
```

* run `uv run tag_teis.py`

### evaluation

```bash
uv run evaluate.py
```

and inspect [llm/stats.csv](llm/stats.csv)