from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)

chat_template = ChatPromptTemplate.from_messages(
    [
        ("system", "Answer the following question preceded by So, yeah."),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

context_prompt = """You are a trusted assistant that helps answer questions based only on the provided context. Here is some context which might or might not help you answer: {context}.  If the context is not helpful, you should say you do not know, and summarize the context in one sentence."""

context_template = ChatPromptTemplate.from_messages(
    [("system", context_prompt), MessagesPlaceholder(variable_name="question")]
)
