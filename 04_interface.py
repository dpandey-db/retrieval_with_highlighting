# Databricks notebook source
# MAGIC %md
# MAGIC # Interface
# MAGIC A serving endpoint abstracts away a lot of the complexities of AI systems - especially when combined with agentic frameworks. We can test our serving endpoint below to see how the input and output signatures will react with our deployed agent.
# MAGIC
# MAGIC This code tests connections to the served endpoint pror to running our streamlit app. It covers custom outputs and why they are important for both apps and agent evaluation.

# COMMAND ----------

# MAGIC %md
# MAGIC Because most interfaces work on an API call, we use the requests package. One of the most important parts of testing the endpoint is determining the output content so we can parse it accordingly.

# COMMAND ----------

import time
import requests
import numpy as np
import mlflow.pyfunc
import json

serving_endpoint_name = "agents_devanshu_pandey-retriever_agent_demo-elaws_retriever"

API_URL = f"https://adb-984752964297111.11.azuredatabricks.net/serving-endpoints/{serving_endpoint_name}/invocations"

API_TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

data = {
    "messages": [{"role": "user", "content": "What regulations cover the erection of temporary structures?"}]
}
headers = {"Content-Type": "application/json", "Authorization": f"Bearer {API_TOKEN}"}
response = requests.post(url=API_URL, json=data, headers=headers)

response.content

# COMMAND ----------

# MAGIC %md
# MAGIC The response has some important fields, namely the response history (under choices -> message --> content), metadata about the call, message history, and a documents section to specificaly return our retrieved documents.

# COMMAND ----------

# response
json.loads(response.content)["choices"][0]["message"]["content"]

# COMMAND ----------

# message history
json.loads(response.content)["custom_outputs"]["message_history"][0]

# COMMAND ----------

# documents
json.loads(response.content)["custom_outputs"]["documents"][0]

# COMMAND ----------

# MAGIC %md
# MAGIC ## How Databricks Apps Work
# MAGIC
# MAGIC Databricks apps are a hosted container service. They use a yaml configuration file (`interface/app.yaml`) to specify a run path where we run a main python file. This pattern is universal for pretty much all front ends written in python. In this accelerator we leverage Streamlit to share the links and information related to the retriever documents. This is highly customizable using general UI frameworks like Streamlit, Gradio, or Flask.
