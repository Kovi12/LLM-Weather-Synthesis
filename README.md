# Weather data synthesis with language models

Bachelor-thesis experiments on adapting language models for weather-data synthesis.

The notebooks cover data collection/preparation, LoRA fine-tuning, forecast generation and scoring. Scripts compare outputs using ROUGE/BLEURT and visualize results. The weather examples combine scraped structured information with language-model prompts.

Use an environment with PyTorch, Transformers, PEFT/TRL, datasets and the packages imported by the selected notebook. Configure local dataset paths, model access and external API credentials before use. `HF_TOKEN` and `REPLICATE_API_TOKEN` should be supplied locally where required.

Notebooks contain source cells without saved outputs. Datasets, access tokens, model weights, personal documents and reference theses are excluded. Base-model and source-data licenses apply separately.

Bachelor project by Ovidiu Catalin Ghibea, supervised by Dan-Matei Popovici.
