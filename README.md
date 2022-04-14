### Credits

For images caption generation OFA-model is used (https://github.com/ofa-sys/ofa).

For word embedding - "all-distilroberta-v1" model (https://huggingface.co/sentence-transformers/all-distilroberta-v1).

Elasticsearch for indexing.

### Project structure

The repository is missing modules "criterions", "data", "fairseq", "models", "tasks" and "utils" from the original OFA repository. Also it is missing directories "dataset" (should contain images to index) and "checkpoints" (in my case it cointains "Finetuned checkpoint for Caption on COCO" checkpoint from the original repository (https://github.com/OFA-Sys/OFA/blob/main/checkpoints.md)).

The presented GUI is incomplete.
