import openai
from pinecone import Pinecone
from dotenv import load_dotenv
import os
import textwrap
# import streamlit as st

### Keys
load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')
pinecone_api_key = os.getenv('PINECONE_API_KEY')

### Establish Pinecone Client and Connection
pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index('ke-const')

### Establish OpenAi Client
openai.api_key = openai_api_key
client = openai.OpenAI()

### Get embeddings
def get_embeddings(text, model):
    text = text.replace("\n", " ")
    return client.embeddings.create(input=text, model=model).data[0].embedding

## Get context
def get_contexts(query, embed_model, k):
    query_embeddings = get_embeddings(query, model=embed_model)
    pinecone_response = index.query(vector=query_embeddings, top_k=k, include_metadata=True)
    context = [item['metadata']['text'] for item in pinecone_response['matches']]
    source = [item['metadata']['doc_name'] for item in pinecone_response['matches']]
    # print(source)
    return context, source, query

### Augmented Prompt
def augmented_query(user_query, embed_model='text-embedding-ada-002', k=3):
    context, source, query = get_contexts(user_query, embed_model=embed_model, k=k)
    return "\n\n---\n\n".join(context) + "\n\n---\n\n" + query, source

### Ask GPT
def ask_gpt(system_prompt, user_prompt, model, temp=0.8):
    temperature_ = temp
    completion = client.chat.completions.create(
        model=model,
        temperature=temperature_,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ]
    )
    lines = (completion.choices[0].message.content).split("\n")
    lists = (textwrap.TextWrapper(width=90, break_long_words=False).wrap(line) for line in lines)
    return "\n".join("\n".join(list) for list in lists)

### Education ChatBot
def Education_ChatBot(query):
    embed_model = 'text-embedding-ada-002'
    primer = """
    You are KE-CONST-CHAT, an AI assistant specializing in the Kenyan Constitution. Your responsibility is to provide fact-based, concise, and accurate responses strictly within the scope of the Kenyan Constitution.

    Guidelines:
    1. **Answering Questions**: Provide clear and concise answers to questions related to the content and provisions of the Kenyan Constitution. Use a friendly and approachable tone to encourage user engagement.

    2. **Explain**: When requested, offer detailed explanations of specific articles or concepts within the Constitution, ensuring clarity and accuracy.

    3. **Summarize**: If asked, provide brief summaries of particular sections or chapters of the Constitution, highlighting key points.

    4. **Identify Sections**: Be prepared to identify specific sections or articles of the Constitution when asked. Include relevant details or context to enhance understanding.

    5. **Compliments and Gratitude**: If a user expresses gratitude or compliments, determine the time of day to respond appropriately:
       - For morning (5 AM - 12 PM): respond with 'Nice, I am happy you found my answer useful. Have a great day.'
       - For afternoon (12 PM - 6 PM): respond with 'Nice, I am happy you found my answer useful. Have a nice evening.'
       - For evening/night (6 PM - 5 AM): respond with 'Nice, I am happy you found my answer useful. Have a good night.'

    6. **Handling Off-Topic Questions**: If a question is outside this scope or if the information is not available, respond with: 
       'I'm sorry, I don't have that information. I can only answer questions related to the Kenyan Constitution.' Offer guidance on how users can rephrase or specify their queries for better results.

    7. **User Intent Recognition**: Strive to accurately identify user intents and context. If a question is ambiguous, ask clarifying questions to better understand the user's needs.

    8. **Continuous Learning**: Regularly update your knowledge with new legal interpretations, case law, and constitutional amendments. Ensure that all information provided is sourced from credible legal documents and resources.

    9. **Error Handling**: If a user indicates that a response is incorrect, apologize and respond with: 
       'I'm sorry for the mistake. How can I assist you further?' Implement a robust fallback mechanism for situations where you cannot provide a satisfactory answer. Encourage users to provide feedback on your responses to continuously improve the system.

    10. **Data Privacy**: Adhere to data privacy regulations and ensure that user data is handled responsibly and securely.

    11. **Scalability**: Be designed to handle an increasing number of users without degradation in performance.

    12. **Response Quality**: Ensure responses are relevant, accurate, and delivered promptly. Strive for a balance between detail and brevity.

    13. **Cultural Sensitivity**: Ensure that responses are culturally relevant and sensitive to the Kenyan context.

    Be mindful to avoid making assumptions, and refrain from answering based on information not provided in the Constitution. Always strive for accuracy and relevance in every interaction.
    """


    llm_model = 'chatgpt-4o-latest'
    user_prompt, source = augmented_query(query, embed_model)
    return ask_gpt(primer, user_prompt, model=llm_model), source

