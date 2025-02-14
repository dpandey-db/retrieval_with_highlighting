<img src=https://raw.githubusercontent.com/databricks-industry-solutions/.github/main/profile/solacc_logo.png width="600px">

# Semantic Legislation Search 

[![DBR](https://img.shields.io/badge/DBR-15.4_LTS_ML-red?logo=databricks&style=for-the-badge)](https://docs.databricks.com/release-notes/runtime/CHANGE_ME.html)
[![CLOUD](https://img.shields.io/badge/CLOUD-AZURE-blue?logo=googlecloud&style=for-the-badge)](https://databricks.com/try-databricks)

## Business Problem
Government agencies have a large corpus of acts and regulations that are updated frequently. These documents are often stored in PDF format and are difficult to search and retrieve. These regulations can be complex and difficult to understand for the average citizen.

## Proposed Solution
This solution accelerator showcases a simple end-to-end search based on vector and semantic ranking. It integrates with a user interface to allow users to search for regulations and get relevant results, with semantic highlighting of the most relevant sections related to the search query. It also showcases the Mosaic AI Agent Evaluation Framework for providing a fast way to bootstrap a solution from scratch with team.

## Reference Architecture
We can divide our solution into three components: ingestion, featurization, and inference. We also use Databricks Apps to provide a basic user inferface for the solution. We leverage Mosaic Vector Search to do retrieval, Mosaic AI Gateway to serve the foundation model, and Mosaic AI Agent Evaluation Framework to serve the agent framework.

<img src="assets/Semantic RAG Architecture.png" width="800px">

## Key Services and Costs

| Service | Example Cost* | Latency | Reference |
|---------|------------|---------------|----------|
| Databricks Apps |  $180/month | <100ms | [Apps Pricing](https://www.databricks.com/product/pricing) |
| Mosaic Vector Search  | $250/month | 10-100ms | [Docs](https://docs.databricks.com/en/generative-ai/vector-search.html) |
| Mosaic AI Gateway  | $1.00/1M tokens | 500-5000ms | [Docs](https://docs.databricks.com/en/machine-learning/ai-gateway/index.html) |
| Mosaic AI Model Serving  | $250/month | ~100ms | [Docs](https://docs.databricks.com/en/machine-learning/model-serving/index.html) |

\* Example costs are illustrative estimates only and will vary based on usage, region, and implementation details. DBU = Databricks Unit.

## Table of Contents

Each solution component is implemented as a Databricks notebook.

1. Ingestion: Ingest the documents from a source and parse them in volumes and delta tables.

2. Featurization: Feature engineer the raw documents into chunks and metadata that are useful for search and retrieval.

3. Inference: Use a foundation model and agent framework to search and extract information from the documents.

4. Interface: Provide a basic user interface for interacting with the agent and gathering feedback.

## Authors
<devanshu.pandey@databricks.com>
<scott.mckean@databricks.com>

## Project support 

Please note the code in this project is provided for your exploration only, and are not formally supported by Databricks with Service Level Agreements (SLAs). They are provided AS-IS and we do not make any guarantees of any kind. Please do not submit a support ticket relating to any issues arising from the use of these projects. The source in this project is provided subject to the Databricks [License](./LICENSE.md). All included or referenced third party libraries are subject to the licenses set forth below.

Any issues discovered through the use of this project should be filed as GitHub Issues on the Repo. They will be reviewed as time permits, but there are no formal SLAs for support.

## License

&copy; 2025 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License [https://databricks.com/db-license-source].  All included or referenced third party libraries are subject to the licenses set forth below.

| library                                | description             | license    | source                                              |
|----------------------------------------|-------------------------|------------|-----------------------------------------------------|
|docling|Document parsing and export|MIT|https://github.com/docling/docling|

## Contributing

We welcome contributions to this project. We happily welcome contributions to this project. We use GitHub Issues to track community reported issues and GitHub Pull Requests for accepting changes.