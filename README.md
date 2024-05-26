Install the packages in `requirements.txt` 

```python
pip install -r requirements.txt
```

Start the docker container for qdrant

```
docker run -p 6333:6333 -p 6334:6334 \
    -v qdrant_storage:/qdrant/storage \
    qdrant/qdrant

```

Also, you need to build the Books database

```python
from books_db import build_database
build_database()

```

Start the streamlit server

```
streamlit run Intro.py
```
