# translate_html.py

Sends eligible visible text and a few attributes to OpenAI or DeepL and writes a new HTML file.

Needs OPENAI_API_KEY and OPENAI_MODEL, or DEEPL_API_KEY. Protected regions and template tokens are skipped. A sentence split by inline tags can come back with different verb forms. It does not rebuild a browser layout.

## Run

Set `OPENAI_API_KEY` and `OPENAI_MODEL`.

```text
python translate_html.py source.html delivery.fr.html --provider openai --lang fr --dir ltr
```

Article: https://htmltranslate.com/blog/translate-html-files-python-beautifulsoup-api
